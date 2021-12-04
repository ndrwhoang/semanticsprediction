import random
import numpy as np
import os
import json
from itertools import chain
from tqdm import tqdm
from typing import List, Dict
import wandb

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.dataset.seq2seq_dataset import collate_fn


class Finetuner:
    def __init__(self, config, model, train_dataset, val_dataset=None, test_dataset=None, checkpoint=None):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        if self.val_dataset == None:
            self.val_dataset = train_dataset
        if self.test_dataset == None:
            self.test_dataste = train_dataset
        
        self.set_seed()
        self.device = torch.device('cuda' if 
                                   torch.cuda.is_available and 
                                   self.config['general'].getboolean('use_gpu') else
                                   'cpu')
        print(f'----------- device : {self.device}')
        self.model.to(self.device)
        if checkpoint is not None:
            model.load_state_dict(torch.load(os.path.join(*checkpoint.split('\\'))))
            print(f'Successfully loaded checkpoint from {checkpoint}')
        self.train_dataloader, self.val_dataloader = self._get_dataloader(self.config)
        self.optimizer, self.lr_scheduler = self._get_optimizer(self.config)
        # self.node_label2id, self.edge_label2id = self._load_label_dict(self.config)
        self.masked_node_idx, self.masked_edge_idx = self._get_mask(self.config)
        assert len(set(self.masked_edge_idx)) == len(self.masked_edge_idx)
        assert len(set(self.masked_node_idx)) == len(self.masked_node_idx)
    
    def set_seed(self):
        self.seed = int(self.config['general']['seed'])
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def _get_optimizer(self, config):
        # AdamW + OneCycleLR
        # Currently using maximum learning rate from old experiment
        # TODO: rerun find maximum learning rate
        # Note: onecycle requires running the full number of epochs, do not use early stopping
        # change to something else depeding on full training run time
        total_steps = len(self.train_dataloader) * int(self.config['training']['n_epoch'])
        model_params = list(self.model.named_parameters())
        no_decay = ['bias']
        optimized_params = [
            {
                'params':[p for n, p in model_params if not any(nd in n for nd in no_decay)], 
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }   
        ]
        optimizer = AdamW(optimized_params, lr=float(config['training']['lr_finetune']))
        lr_scheduler = OneCycleLR(
            optimizer, 
            max_lr=float(self.config['training']['max_lr']), 
            total_steps=total_steps
            )
        
        return optimizer, lr_scheduler
    
    def _get_dataloader(self, config):
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=int(config['training']['bsz_finetune']),
                                      collate_fn=collate_fn,
                                      shuffle=False,
                                      drop_last=False)
        val_dataloader = DataLoader(self.val_dataset,
                                    batch_size=int(config['training']['bsz_val']),
                                    collate_fn=collate_fn,
                                    shuffle=False,
                                    drop_last=False)
        
        return train_dataloader, val_dataloader
        
    def _get_mask(self, config):
        # Produce mask to exclude semantic subspaces, 
        # change in config 'training' section
        masked_node_subspace = config['training']['node_subspace'].split(' ')
        masked_edge_subspace = config['training']['edge_subspace'].split(' ')
        if masked_node_subspace == ['none'] and masked_edge_subspace == ['none']:
            return [], []
        subspace_dict_path = os.path.join(*config['data_path']['subspace_dict'].split('\\'))
        with open(subspace_dict_path, 'r') as f:
            subspace2id = json.load(f)
        f.close()
        masked_node_idx = [subspace2id[subspace] for subspace in masked_node_subspace]
        masked_edge_idx = [subspace2id[subspace] for subspace in masked_edge_subspace]
        masked_node_idx = list(chain.from_iterable(masked_node_idx))
        masked_edge_idx = list(chain.from_iterable(masked_edge_idx))
        
        return masked_node_idx, masked_edge_idx   
    
    def _to_device(self, batch):
        out = []
        for item in batch:
            try:
                out.append(item.to(self.device))
            except:
                out.append(item)
        out = tuple(out)
        
        return out
    
    def _calculate_masked_loss(self, pred, true, mask):
        # note: nan * False = nan, not 0
        # mask is subspace mask, mask_ is target value nan mask ('normal' mask)
        mask_ = torch.isnan(true) != True
        # print(mask_)
        # print(mask)
        mask_ = mask_*mask
        # print(mask_)
        loss = torch.sum((torch.nan_to_num(pred-true)**2)*mask_) / (torch.sum(mask_)+0.000001)
        return loss
    
    def _extract_reverse_masks(self, labels, subspace):
        mask = torch.zeros(labels.size())
        if subspace == 'nodes':
            mask[:, :, self.masked_node_idx] = 1.
        elif subspace == 'edges':
            mask[:, :, self.masked_edge_idx] = 1.
        else:
            print('error in choosing subspace to mask')
        mask = mask.to(self.device)
        
        return mask
    
    def run_partial_train(self, run_name: str, finetune_thresh):
        n_sample_trained = 0
        self.run_validation()
        
        if finetune_thresh < 1:
            n_finetune_thresh = float(finetune_thresh) * self.train_dataset.n_sample
        else:
            n_finetune_thresh = finetune_thresh
            
        print(f'Commences finetuning, stopping after {n_finetune_thresh}')
        
        for epoch in range(1):
            self.model.train()
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), mininterval=2)
            total_loss = 0
            for i, batch in pbar:
                batch = self._to_device(batch)
                (_, _, node_labels, _, edge_labels) = batch
                
                # early stopping at proportion of finetune subspace
                n_sample_trained += int(self.config['training']['bsz_finetune'])
                
                if n_sample_trained > n_finetune_thresh:
                    break
                
                # forward
                node_output, edge_output = self.model(batch)
                assert node_output.size() == node_labels.size()
                assert edge_output.size() == edge_labels.size()
                
                node_mask = self._extract_reverse_masks(node_labels, subspace='nodes')
                edge_mask = self._extract_reverse_masks(edge_labels, subspace='edges')
                
                # print(node_mask)
                # print(edge_mask)
                
                node_loss = self._calculate_masked_loss(node_output, node_labels, node_mask)
                edge_loss = self._calculate_masked_loss(edge_output, edge_labels, edge_mask)
                loss = node_loss + edge_loss
                with torch.no_grad():
                    total_loss += loss
                
                # step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.lr_scheduler.step()
                
                # log
                # wandb.log({
                #     'train_total_loss': loss,
                #     'train_node_loss': node_loss,
                #     'train_edge_loss': edge_loss,
                #     'learning_rate': float(self.optimizer.param_groups[0]['lr'])
                #     })
                pbar.set_description(f'(Training) Epoch: 0 - Steps: {i}/{len(self.train_dataloader)} - Loss: {loss}', refresh=True)
            
            print(f'Training loss: {total_loss}')
            self.run_validation()
               
    def run_validation(self):
        pbar = tqdm(enumerate(self.val_dataloader), total = len(self.val_dataloader))
        self.model.eval()
        total_val_loss = 0
        
        for i, batch in pbar:
            batch = self._to_device(batch)
            (_, _, node_labels, _, edge_labels) = batch
            
            # forward
            with torch.no_grad():                
                node_output, edge_output = self.model(batch)
                
                node_mask = self._extract_reverse_masks(node_labels, subspace='nodes')
                edge_mask = self._extract_reverse_masks(edge_labels, subspace='edges')
                
                node_loss = self._calculate_masked_loss(node_output, node_labels, node_mask)
                edge_loss = self._calculate_masked_loss(edge_output, edge_labels, edge_mask)
                val_loss = node_loss + edge_loss
                
                total_val_loss += val_loss

            pbar.set_description(f'(Validating) Steps: {i}/{len(self.val_dataloader)} - Loss: {val_loss}', refresh=True)
                # bs = int(self.config['training']['bsz_val'])
            # wandb.log({
            #         'val_total_loss': val_loss,
            #         'val_node_loss': node_loss,
            #         'val_edge_loss': edge_loss
            #         })
            
        print(f'Validation loss: {total_val_loss}')
        # wandb.log({'epoch_val_loss': total_val_loss})
        
        return total_val_loss
    
def trainer_finetune_test(config):
    from transformers import RobertaTokenizerFast
    
    from src.dataset.seq2seq_dataset import UDSDataset
    from src.model.baseline import BaseModel
    
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    dataset = UDSDataset(config, 'train_subset', tokenizer)
    model = BaseModel(config)
    
    trainer = Finetuner(config, model, dataset)
    trainer.run_partial_train('testtesttest', 5)

if __name__ == '__main__':
    import configparser
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    trainer_finetune_test(config)