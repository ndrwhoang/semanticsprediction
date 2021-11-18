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


class Trainer:
    def __init__(self, config, model, train_dataset, val_dataset=None, test_dataset=None):
        # Base class for training, will also handle inference if inference is simple enough
        # TODO: inference
        # TODO: result tracking
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
        self.train_dataloader, self.val_dataloader = self._get_dataloader(self.config)
        self.optimizer, self.lr_scheduler = self._get_optimizer(self.config)
        # self.node_label2id, self.edge_label2id = self._load_label_dict(self.config)
        self.masked_node_idx, self.masked_edge_idx = self._get_mask(self.config)
    
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
        optimizer = AdamW(optimized_params, lr=float(config['training']['lr']))
        lr_scheduler = OneCycleLR(
            optimizer, 
            max_lr=float(self.config['training']['max_lr']), 
            total_steps=total_steps
            )
        
        return optimizer, lr_scheduler
    
    def _get_dataloader(self, config):
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=int(config['training']['bsz_train']),
                                      collate_fn=collate_fn,
                                      shuffle=False,
                                      drop_last=False)
        val_dataloader = DataLoader(self.val_dataset,
                                    batch_size=int(config['training']['bsz_val']),
                                    collate_fn=collate_fn,
                                    shuffle=False,
                                    drop_last=False)
        
        return train_dataloader, val_dataloader
    
    def _load_label_dict(self, config):
        # Get the semantic subspace id to produce mask
        node_dict_path = os.path.join(*config['data_path']['node_dict'].split('\\'))
        edge_dict_path = os.path.join(*config['data_path']['edge_dict'].split('\\'))
        
        with open(node_dict_path, 'r') as f:
            node_label2id = json.load(f)
        f.close()
        with open(edge_dict_path, 'r') as f:
            edge_label2id = json.load(f)
        f.close()
        
        return node_label2id, edge_label2id
        
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
        loss = torch.sum((torch.nan_to_num(pred-true)*mask)**2.0) / torch.sum(mask)
        
        return loss
    
    def ______process_labels(self, labels):
        # For old data pipeline, keep for posterity
        label_out = []
        for i_sample, label in enumerate(labels):
            label_tensor = [torch.tensor(label_vec, dtype=torch.float) for label_vec in label.values()]
            label_tensor = torch.stack(label_tensor, dim=0)
            label_out.append(label_tensor)
        label_out = pad_sequence(label_out, batch_first=True)
        
        mask = labels != float('nan')
        mask = mask.to(self.device)
        
        return labels, mask
    
    def _extract_masks(self, labels, subspace):
        # mask = labels != float('nan')
        mask = torch.ones(labels.size())
        if subspace == 'nodes':
            mask[:, self.masked_node_idx] = 0.
        elif subspace == 'edges':
            mask[:, self.masked_edge_idx] = 0.
        else:
            print('error in choosing subspace to mask')
        mask = mask.to(self.device)
        
        return mask
    
    def run_train(self, run_name: str):
        # Take run_name to be used in saved checkpoint
        # Save checkpoint everytime val_loss decreases
        # Note: tqdm(enumerate) cause memory leakage?
        # TODO: set up proper experiment tracking
        total_train_step = 0
        
        for epoch in range(int(self.config['training']['n_epoch'])):
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), mininterval=2)
            self.model.train()
            best_loss = float('inf')
            
            for i, batch in pbar:
                # if i == 5: break
                batch = self._to_device(batch)
                (_, _, node_labels, _, edge_labels) = batch
                
                # forward
                node_output, edge_output = self.model(batch)
                assert node_output.size() == node_labels.size()
                assert edge_output.size() == edge_labels.size()
                # print(node_output[0])
                # print(node_labels[0])
                
                node_mask = self._extract_masks(node_labels, subspace='nodes')
                edge_mask = self._extract_masks(edge_labels, subspace='edges')
                
                node_loss = self._calculate_masked_loss(node_output, node_labels, node_mask)
                edge_loss = self._calculate_masked_loss(edge_output, edge_labels, edge_mask)
                loss = node_loss + edge_loss

                # step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                
                # log
                total_train_step += 1
                bs = int(self.config['training']['bsz_train'])
                wandb.log({
                    'train_total_loss': loss/bs,
                    'train_node_loss': node_loss/bs,
                    'train_edge_loss': edge_loss/bs,
                    'learning_rate': self.lr_scheduler.get_last_lr()
                    })
                pbar.set_description(f'(Training) Epoch: {epoch} - Steps: {i}/{len(self.train_dataloader)} - Loss: {loss}', refresh=True)

            val_loss = self.run_validation()
            if val_loss < best_loss:
                best_loss = val_loss
                self._save_model(self.model, run_name + '_' + self.config['model_path']['base_model'])
            
    def run_validation(self):
        pbar = tqdm(enumerate(self.val_dataloader), total = len(self.val_dataloader))
        self.model.eval()
        val_loss = 0
        
        for i, batch in pbar:
            batch = self._to_device(batch)
            (_, node_ids, node_labels, edge_ids, edge_labels) = batch
            
            # forward                
            node_output, edge_output = self.model(batch)
            
            node_mask = self._extract_masks(node_labels, subspace='nodes')
            edge_mask = self._extract_masks(edge_labels, subspace='edges')
            
            node_loss = self._calculate_masked_loss(node_output, node_labels, node_mask)
            edge_loss = self._calculate_masked_loss(edge_output, edge_labels, edge_mask)
            loss = node_loss + edge_loss
            
            val_loss += loss

            pbar.set_description(f'(Validating) Steps: {i}/{len(self.val_dataloader)} - Loss: {loss}', refresh=True)
            # bs = int(self.config['training']['bsz_val'])
            # wandb.log({
            #         'val_total_loss': loss/bs,
            #         'val_node_loss': node_loss/bs,
            #         'val_edge_loss': edge_loss/bs
            #         })
            
        print(f'Validation loss: {val_loss}')
        wandb.log({'epoch_val_loss': val_loss})
        
        return val_loss
    
    def _save_model(self, model, path):
        save_path = os.path.join(*path.split('\\'))
        torch.save(model.state_dict(), open(save_path, 'wb'))
    
def trainer_test(config):
    from transformers import RobertaTokenizerFast
    
    from src.dataset.seq2seq_dataset import UDSDataset
    from src.model.baseline import BaseModel
    
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    dataset = UDSDataset(config, 'train', tokenizer)
    model = BaseModel(config)
    
    trainer = Trainer(config, model, dataset)
    trainer.run_train('testtesttest')


if __name__ == '__main__':
    import configparser
    
    # config = configparser.ConfigParser()
    # config.read(os.path.join('configs', 'config.cfg'))
    
    # trainer_test(config)