"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import os
import math
import logging
import pdb

import wandb

from tqdm import tqdm
import numpy as np

import torch
from torch.nn import functional as F

from torch.utils.data.dataloader import DataLoader

from utils import sample

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_iters = 0
    final_iters = 0  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_dir = None
    samples_dir = None
    sample_every = 1
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Eval:
    def __init__(self, model, test_dataset, config):
        self.model = model
        self.test_dataset = test_dataset
        self.config = config
        self.device=config.device
        self.model.to(device=self.device)   

    def calculate_coverage(self, layout):
        L_layout = np.asarray(layout.convert("L"))
        color_white = np.where(L_layout == 255)[0].shape[0]
        return color_white

    def eval(self, command):
        model = self.model
        raw_model = model.module if hasattr(self.model, "module") else model
        if self.config.model_path != None:
            raw_model.load_state_dict(torch.load(self.config.model_path))
        else:
            print("args model_path is None")
            return
        loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)
        print("dataset length:", len(loader))

        pbar = tqdm(enumerate(loader), total=len(loader))
        results = []
        total_box_num = 0
        color_white_num = 0
                
        with torch.no_grad():         
            for it, (x, y) in pbar:
                x_cond = x.to(self.device)
                
                if command["name"] == "random_generate":    
                    layouts = sample(model, x_cond[:, :1], steps=self.test_dataset.max_length,
                                temperature=1.0, sample=True, top_k=5, only_label=False, gt=x_cond).detach().cpu().numpy()
                elif command["name"] == "category_generate":
                    layouts = sample(model, x_cond[:, :1], steps=self.test_dataset.max_length,
                                temperature=1.0, sample=True, top_k=5, only_label=True, gt=x_cond).detach().cpu().numpy()
                elif command["name"] == "real_image":
                    layouts = x_cond.detach().cpu().numpy()
                elif command["name"] == "reconstruction":
                    logits, _ = model(x_cond)
                    probs = F.softmax(logits, dim=-1)
                    _, y = torch.topk(probs, k=1, dim=-1)
                    layouts = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
                else:
                    raise ValueError(f"{command['name']} dose not exist.")                    

                for i, layout in enumerate(layouts):
                    if command["save_image"]:
                        layout_img = self.test_dataset.render(layout)
                        layout_img.save(os.path.join(self.config.samples_dir, f'{command["name"]}_{it:02d}_{i:02d}.png'))

                    if command["calculate_coverage"]:
                        layout_img = self.test_dataset.render(layout)
                        color_white = self.calculate_coverage(layout_img)
                        total_box_num += 1
                        color_white_num += color_white

                    if command["save_pkl"]:
                        box_and_label = self.test_dataset.render_normalized_layout(layout)
                        results.append(box_and_label)
                     

        if command["calculate_coverage"]:
            Height = self.test_dataset.H
            Width = self.test_dataset.W
            coverage_rate = color_white_num / (total_box_num * Height * Width)
            print("coverage rate:", coverage_rate)

        if command["save_pkl"]:
            import pickle
            with open(self.config.evaluate_layout_path, 'wb') as fb:
                pickle.dump(results, fb)
        

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.iters = 0
        self.fixed_x = None
        self.fixed_y = None
        self.device=config.device
        self.model.to(device=self.device)
        self.inference_iter = config.inference_iter

        # print("Using wandb")
        # wandb.init(project='LayoutTransformer', name=config.exp,entity="goosefarm")
        # wandb.config.update(config)

        # take over whatever gpus are on the system
        # self.device = 'cpu'
        # if torch.cuda.is_available():
        #     self.device = torch.cuda.current_device()
        #     self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, 'checkpoint.pth')
        logger.info("saving %s", ckpt_path)
        torch.save(raw_model.state_dict(), ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        if self.config.model_path != None:
            raw_model.load_state_dict(torch.load(self.config.model_path))

        optimizer = raw_model.configure_optimizers(config)
        pad_token = self.train_dataset.vocab_size - 1

        def gen_batch(data_b):#fatgoose
            tokens = data_b['text'].long().to(self.device)
            # types = data_b['types'].long()
            # sentence_order = data_b['is_random'].long()
            input_mask = data_b['input_mask'].float().to(self.device)
            loss_mask = data_b['loss_mask'].float().to(self.device)
            lm_labels = data_b['lm_output'].long().to(self.device)
            targets = data_b['target_output'].long().to(self.device)
            element_num_list = data_b['element_num'].long().to(self.device)
            # padding_mask = data_b['padding_mask'].long()
            # print(input_mask)
            # print(element_num_list)
            # pdb.set_trace()

            # return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask
            return tokens,targets,lm_labels,input_mask,loss_mask,element_num_list


        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, data_batch in pbar:

                # complete
                # pdb.set_trace()
                if epoch == 0 and not is_train:
                    self.fixed_x = data_batch['text'][:min(4, data_batch['text'].size(0))].to(self.device)
                    self.fixed_y = data_batch['target_output'][:min(4, data_batch['target_output'].size(0))].to(self.device)

                # place data on the correct device
                # data_batch = data_batch.to(self.device)

                #fatgoose BLT
                tokens,targets,lm_labels,input_mask,loss_mask,element_num_list = gen_batch(data_batch)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # import ipdb; ipdb.set_trace()

                    #logits, loss = model(idx=x, targets=y, pad_token=pad_token)
                    if is_train:
                        logits, loss = model(idx=tokens, targets=(targets, lm_labels))
                    else:
                        logits, loss = model(idx=tokens, targets=(targets, lm_labels))
                        # logits, loss = model.inference(tokens, self.inference_iter, element_num_list,mask_token=input_mask,targets=targets)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    self.iters += 1
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        # self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.iters < config.warmup_iters:
                            # linear warmup
                            lr_mult = float(self.iters) / float(max(1, config.warmup_iters))
                        else:
                            # cosine learning rate decay
                            progress = float(self.iters - config.warmup_iters) / float(max(1, config.final_iters - config.warmup_iters))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    # wandb.log({
                    #     'train loss': loss.item(),
                    #     'lr': lr, 'epoch': epoch+1
                    # }, step=self.iters)
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                # wandb.log({'test loss': test_loss}, step=self.iters)
                return test_loss

        best_loss = float('inf')
        for epoch in range(config.max_epochs):
            run_epoch('train')
            if self.test_dataset is not None:
                with torch.no_grad():
                    test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_dir is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()

            # sample from the model
            if self.config.samples_dir is not None and (epoch+1) % self.config.sample_every == 0:
                
                if (epoch+1) == self.config.sample_every:
                    layouts = self.fixed_x.detach().cpu().numpy()
                    for i, layout in enumerate(layouts):
                        layout = self.train_dataset.render(layout)
                        layout.save(os.path.join(self.config.samples_dir, f'input_{i:02d}.png'))
                
                layouts = sample(model, self.fixed_x[:, :1], steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=True, top_k=5, only_label=True, gt=self.fixed_x).detach().cpu().numpy()
                for i, layout in enumerate(layouts):
                    layout = self.train_dataset.render(layout)
                    layout.save(os.path.join(self.config.samples_dir, f'category_{epoch:02d}_{i:02d}.png'))