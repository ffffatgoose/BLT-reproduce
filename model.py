"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F

from copy import deepcopy

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        #fatgoose what???
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None,mask = None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #TODO
        if mask is not None:#fatgoose
            att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))

        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        #?
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x,mask = None):
        x = x + self.attn(self.ln1(x),mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, pad_token=-100):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # x = self.drop(token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token)

        return logits, loss

class BLT_encoder(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd)) #TODO
        # self.pos_emb = PositionalEmbedding(d_model=config.n_embd)


        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        self.vocab_size = config.vocab_size
        self.criterion = nn.NLLLoss(ignore_index=0)#self.vocab_size - 4) #fatgoose
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)


        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, pad_token=-100):


        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (idx != self.vocab_size - 1).unsqueeze(1).repeat(1, idx.size(1), 1).unsqueeze(1) #TODO catogory mask

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # x = self.drop(token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        #fatgoose
        logsoftmax_logits = self.logsoftmax(logits)

        # pdb.set_trace()

        # if we are given some desired targets also calculate the loss
        #TODO mask loss
        loss = None

        if targets is not None:
            # target_output = targets[0]
            lm_output = targets[1]
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token)
            loss = self.criterion(logsoftmax_logits.view(lm_output.size(0),self.vocab_size,-1),lm_output) #fatgoose
            # pdb.set_trace()
            # # test criterion correct
            # preds = F.one_hot(targets[0],num_classes=self.vocab_size)
            # lm_output_tmp = F.one_hot(lm_output,num_classes=self.vocab_size)
            # print(preds.size())
            # loss_origin = self.criterion(lm_output_tmp.view(lm_output.size(0),self.vocab_size,-1).float(), lm_output)  # fatgoose
            # loss = self.criterion(preds.view(lm_output.size(0), self.vocab_size, -1).float(), lm_output)
            # pdb.set_trace()
        else:
            softmax_logits = self.softmax(logits)
            results = softmax_logits.max(dim=-1) # batchSize * maxLength
            seq_new_q = results[0]
            seq_new = results[1]

            # pdb.set_trace()

            return seq_new,seq_new_q,logsoftmax_logits

        return logits, loss
    
    def filter_top_k(self,seq_new_q, seq_new_l,tmp_mask, g,r_i,element_num_list,mask_id):
        # pdb.set_trace()
        if g == 0:#C
            n_i = (r_i * element_num_list //5 ).to(torch.int64).to(torch.cuda.current_device())
            batchs,length = seq_new_q.size()
            group_q = torch.ones((batchs,length//5)).to(torch.cuda.current_device())
            for i in range(seq_new_q.size(1)//5):
                group_q[:,i] = seq_new_q[:,5*i]

            # pdb.set_trace()

            for i in range(batchs):
                group_q[i,int(element_num_list[i]/5):] = 1

            # pdb.set_trace()

            new_mask = deepcopy(tmp_mask).to(torch.cuda.current_device())
            for i in range(seq_new_q.size(1)//5):
                new_mask[:,i*5] = False
            seq_new_masked = deepcopy(seq_new_l).to(torch.cuda.current_device())
            masked_position_origin = torch.argsort(group_q,dim=-1) * 5
            # pdb.set_trace()
            for j in range(batchs):
                # masked_position_origin = torch.topk(group_q[j], n_i[j], dim=-1, largest=False)[1] * 5
               if n_i[j] != 0:
                   new_mask_tmp = masked_position_origin[j, :n_i[j]]
                   new_mask[j][new_mask_tmp] = True
                   seq_new_masked[j][new_mask_tmp] = mask_id
                # pdb.set_trace()

            return seq_new_masked,new_mask


        elif g == 1: # P

            # n_i = (r_i * element_num_list * 2).to(torch.int64)
            n_i = (r_i * (element_num_list // 5) * 2 ).to(torch.int64).to(torch.cuda.current_device())
            batchs, length = seq_new_q.size()
            group_q = torch.ones((batchs, length // 5 * 2)).to(torch.cuda.current_device())
            for i in range(seq_new_q.size(1) // 5):
                group_q[:, 2* i] = seq_new_q[:, 5 * i + 1]
                group_q[:, 2 * i + 1] = seq_new_q[:, 5 * i + 2]

            # pdb.set_trace()
            # ignore padding
            for i in range(batchs):
                group_q[i, int(element_num_list[i] / 5 * 2):] = 1

            pdb.set_trace()

            new_mask = deepcopy(tmp_mask).to(torch.cuda.current_device())
            for i in range(seq_new_q.size(1) // 5):
                new_mask[:, i * 5 + 1] = False
                new_mask[:, i * 5 + 2] = False
            seq_new_masked = deepcopy(seq_new_l).to(torch.cuda.current_device())
            masked_position_origin = torch.argsort(group_q, dim=-1)
            # pdb.set_trace()
            for j in range(batchs):
                # masked_position_origin = torch.topk(group_q[j], n_i[j], dim=-1, largest=False)[1] * 5
                if n_i[j] != 0:
                    new_mask_tmp = masked_position_origin[j, :n_i[j]]
                    for idx_i in range(len(new_mask_tmp)):
                        if new_mask_tmp[idx_i] %2 == 0:
                            new_mask_tmp[idx_i] = new_mask_tmp[idx_i] // 2 * 5 + 1
                        else:
                            new_mask_tmp[idx_i] = new_mask_tmp[idx_i] // 2 * 5 + 2
                    new_mask[j][new_mask_tmp] = True
                    seq_new_masked[j][new_mask_tmp] = mask_id
                # pdb.set_trace()

            return seq_new_masked, new_mask
        else: #S
            # n_i = (r_i * element_num_list * 2).to(torch.int64)
            n_i = (r_i * (element_num_list // 5) * 2 ).to(torch.int64).to(torch.cuda.current_device())
            batchs, length = seq_new_q.size()
            group_q = torch.ones((batchs, length // 5 * 2)).to(torch.cuda.current_device())
            for i in range(seq_new_q.size(1) // 5):
                group_q[:, 2 * i] = seq_new_q[:, 5 * i + 3]
                group_q[:, 2 * i + 1] = seq_new_q[:, 5 * i + 4]

            # pdb.set_trace()
            # ignore padding
            for i in range(batchs):
                group_q[i, int(element_num_list[i] / 5 * 2):] = 1

            # pdb.set_trace()

            new_mask = deepcopy(tmp_mask).to(torch.cuda.current_device())
            for i in range(seq_new_q.size(1) // 5):
                new_mask[:, i * 5 + 3] = False
                new_mask[:, i * 5 + 4] = False
            seq_new_masked = deepcopy(seq_new_l).to(torch.cuda.current_device())
            masked_position_origin = torch.argsort(group_q, dim=-1)
            # pdb.set_trace()
            for j in range(batchs):
                # masked_position_origin = torch.topk(group_q[j], n_i[j], dim=-1, largest=False)[1] * 5
                if n_i[j] != 0:
                    new_mask_tmp = deepcopy(masked_position_origin[j, :n_i[j]]).to(torch.cuda.current_device())
                    for idx_i in range(len(new_mask_tmp)):
                        if new_mask_tmp[idx_i] % 2 == 0:
                            new_mask_tmp[idx_i] = new_mask_tmp[idx_i] // 2 * 5 + 3
                        else:
                            new_mask_tmp[idx_i] = new_mask_tmp[idx_i] // 2 * 5 + 4
                        if new_mask_tmp[idx_i]>= element_num_list[j]:
                            print(new_mask_tmp[idx_i],element_num_list[j])
                            pdb.set_trace()
                    new_mask[j][new_mask_tmp] = True
                    seq_new_masked[j][new_mask_tmp] = mask_id
                # pdb.set_trace()

            return seq_new_masked, new_mask

        # get target group output P and position


    
    def mask_seq(self,seq_l_new,new_mask):
        return None
    
    def inference(self, seq_l, T, element_num_list,mask_token, targets=None, pad_token=-100):#fatgoose seq_l:idxs list
        group_order_list = [0,2,1]#CSP
        for g in group_order_list:
            for i in range(int(T/3)):
                seq_l_new,seq_new_q,new_logits = self.forward(seq_l)
                # seq_l_new = torch.max(seq_new_q,dim=-1)[1] #size right?? #fatgoose
                r_i = float(T - 3 * i - 3) / T #* len(seq_l)
                seq_l_mask, filter_mask = self.filter_top_k(seq_new_q, seq_l_new, mask_token, g, r_i, element_num_list,
                                                            mask_id=self.vocab_size - 4)

                # pdb.set_trace()

                seq_l = seq_l_mask

                # if r_i != 0:
                #     seq_l_mask,filter_mask = self.filter_top_k(seq_new_q,seq_l_new,mask_token,g,r_i,element_num_list,mask_id=self.vocab_size -4)
                #     # pdb.set_trace()
                #     seq_l = seq_l_mask
                # else:
                #     # add mask
                #     seq_l = seq_l_new

        # pdb.set_trace()
        new_loss = self.criterion(new_logits.view(targets[0].size(0),self.vocab_size,-1),targets)

        return seq_l,new_loss


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]