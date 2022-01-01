import random
import numpy as np
import os
import json
from itertools import chain
from tqdm import tqdm
from typing import List, Dict
import wandb
import logging

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler

# from src.dataset.seq2seq_dataset import collate_fn

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Trainer:
    def __init__(self, config, model, train_dataset, val_dataset=None, test_dataset=None, checkpoint=None):
        # TODO: set deepspeed (doesnt work on Windows?)
        # Base class for training
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        # Default to using train set if validation set not provided
        if self.val_dataset == None:
            self.val_dataset = train_dataset
        if self.test_dataset == None:
            self.test_dataste = train_dataset
        
        # Seed
        self.set_seed()
        
        # Use GPU if available
        self.device = torch.device('cuda' if 
                                   torch.cuda.is_available and 
                                   self.config['general'].getboolean('use_gpu') else
                                   'cpu')
        logger.info(f'----------- device : {self.device}')
        
        # Load checkpoint if available
        self.model.to(self.device)
        if checkpoint is not None:
            model.load_state_dict(torch.load(os.path.join(*checkpoint.split('\\'))))
            
        # Create dataloaders
        self.train_dataloader, self.val_dataloader = self._get_dataloader(self.config)
        
        # Create optimizers
        self.optimizer, self.lr_scheduler = self._get_optimizer(self.config)
        
        # Mixed precision training
        self.use_amp = self.config['training'].getboolean('mixed_precision')
        if self.use_amp:
            self.grad_scaler = GradScaler()
            
        # Gradient accumulation
        self.use_grad_accumulation = self.config['training'].getboolean('grad_accumulation')
        if self.grad_accum:
            self.grad_accumulation_steps = float(self.config['training']['grad_accumulation_steps'])
            
    
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
                                      collate_fn=self.train_dataset.collate_fn,
                                      shuffle=False,
                                      drop_last=True,
                                      num_workers=int(config['general']['num_worker']),
                                      pin_memory=True)
        val_dataloader = DataLoader(self.val_dataset,
                                    batch_size=int(config['training']['bsz_val']),
                                    collate_fn=self.val_dataset.collate_fn,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=int(config['general']['num_worker']),
                                    pin_memory=True)
        
        return train_dataloader, val_dataloader
    
    def _to_device(self, batch):
        for k, v in batch.items():
            try:
                batch[k] = v.to(self.device)
            except:
                batch[k] = v       
        
        return batch
    
    def _calculate_masked_loss(self, pred, true):
        # note: nan * False = nan, not 0
        # mask is subspace mask, mask_ is target value nan mask ('normal' mask)
        mask_ = torch.isnan(true) != True
        loss = torch.sum((torch.nan_to_num(pred-true)**2)*mask_) / (torch.sum(mask_)+0.000001)
        return loss

    def run_partial_train(self, run_name: str, finetune_thresh):
        self.model.train()
        pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), mininterval=2)
        epoch_loss = 0
        batch_loss = 0
        self.model.zero_grad(set_to_none=True)
        
        n_sample_trained = 0
        n_finetune_thresh = float(finetune_thresh) * len(self.train_dataloader)
        bs = int(self.config['training']['bsz_train'])
        
        for i, batch in pbar:
            # early stopping at proportion of finetune subspace
            n_sample_trained += bs
            if n_sample_trained > n_finetune_thresh:
                break
            
            batch = self._to_device(batch)
            batch_loss = self._training_step(batch)  
                
            # step
            self.optimizer.step()
            self.model.zero_grad(set_to_none=True)
            
            # log
            pbar.set_description(f'(Training) Epoch: 0 - Steps: {i}/{len(self.train_dataloader)} - Loss: {batch_loss}', refresh=True)
            epoch_loss += batch_loss
            batch_loss = 0
            
        logger.info(f'Training loss: {epoch_loss}')
        val_loss = self.run_validation()
        logger.info(val_loss)
      
    def run_train(self, run_name: str):
        # best_loss = self.run_validation()
        best_score = float('inf')
        
        for epoch in range(int(self.config['training']['n_epoch'])):
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), mininterval=2)
            epoch_loss = 0
            batch_loss = 0
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            
            for i, batch in pbar:
                # if i == 10: break
                batch = self._to_device(batch)
                batch_loss = self._training_step(batch)  
                 
                # step
                if (i+1) % self.grad_accumulation_steps == 0 or self.use_grad_accumulation == False:
                    if not self.use_amp:
                        self.optimizer.step()
                    else:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    self.model.zero_grad(set_to_none=True)
                
                pbar.set_description(f'(Training) Epoch: {epoch} - Steps: {i}/{len(self.train_dataloader)} - Loss: {batch_loss}', refresh=True)
                epoch_loss += batch_loss
                batch_loss = 0
                
            val_loss = self.run_validation()
            best_score = self._save_ckpt_unfreeze_on_eval_score(val_loss, best_score, epoch, run_name)
    
    def run_validation(self):
        # TODO: adds correlation as a metric 
        pbar = tqdm(enumerate(self.val_dataloader), total = len(self.val_dataloader))
        self.model.eval()
        epoch_loss = 0
        batch_loss = 0
        
        for i, batch in pbar:
            batch = self._to_device(batch)
            batch_loss = self._prediction_step(batch)

            pbar.set_description(f'(Validating) Steps: {i}/{len(self.val_dataloader)} - Loss: {batch_loss}', refresh=True)
            epoch_loss += batch_loss
            batch_loss = 0
                        
        logger.info(f'Validation loss: {epoch_loss}')
        wandb.log({'epoch_val_loss': epoch_loss})
        
        return epoch_loss
    
    def _training_step(self, batch):
        self.model.train()
        node_labels = batch['node_labels']
        edge_labels = batch['edge_labels']
        
        # Forward
        node_loss, edge_loss = self.model(batch, return_type='output')
        
        # assert node_loss.size() == node_labels.size()
        # assert edge_loss.size() == edge_labels.size()
        
        masked_node_loss = self._calculate_masked_loss(node_loss, node_labels)
        masked_edge_loss = self._calculate_masked_loss(edge_loss, edge_labels)
        
        logger.debug('==================================')
        logger.debug(node_labels.tolist())
        logger.debug('*')
        logger.debug(node_loss.tolist())
        logging.debug(masked_node_loss.tolist())
        logging.debug(masked_edge_loss.tolist())
        
        loss = masked_node_loss + masked_edge_loss        
        loss.backward()
        
        if self.use_grad_accumulation:
            loss = loss / self.grad_accumulation_steps
        
        # log
        wandb.log({
            'train_total_loss': loss.item(),
            'train_node_loss': masked_node_loss.item(),
            'train_edge_loss': masked_edge_loss.item(),
            'learning_rate': float(self.optimizer.param_groups[0]['lr'])
            })
        
        return loss.item()
    
    @torch.no_grad()
    def _prediction_step(self, batch):
        self.model.valid()
        node_labels = batch['node_labels']
        edge_labels = batch['edge_labels']
        
        # Forward
        node_loss, edge_loss = self.model(batch, return_type='output')
        
        masked_node_loss = self._calculate_masked_loss(node_loss, node_labels)
        masked_edge_loss = self._calculate_masked_loss(edge_loss, edge_labels)
        
        loss = masked_node_loss + masked_edge_loss  
        
        wandb.log({
            'pred_total_loss': loss.item(),
            'pred_node_loss': masked_node_loss.item(),
            'pred_edge_loss': masked_edge_loss.item()
            })  
        
        return loss.item()
    
    def _save_ckpt_unfreeze_on_eval_score(self, val_score, best_score, epoch, run_name):
        if val_score < best_score:
                best_score = val_score
                self._save_model(self.model, self.config['model_path']['checkpoint_dir'] + run_name + '.pt')
                
        elif val_score >= best_score and self.model.frozen == True:
            print(f'Unfreeze encoder at epoch {epoch}')
            self.model.frozen = False
            for g in self.optimizer.param_groups:
                g['lr'] = float(self.config['training']['unfreeze_lr'])
            for param in self.model.pretrained_encoder.parameters():
                param.requires_grad = True
        
        return best_score
    
    def _save_model(self, model, path):
        logger.info(f'Saving model checkpoint at {path}')
        save_path = os.path.join(*path.split('\\'))
        torch.save(model.state_dict(), open(save_path, 'wb'))
    
def trainer_test(config):
    from transformers import RobertaTokenizerFast
    
    from src.dataset.seq2seq_dataset import UDSDataset
    from src.model.baseline import BaseModel
    from src.model.pretrained_roberta import PretrainedModel
    
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    dataset = UDSDataset(config, 'train_subset', tokenizer)
    model = PretrainedModel(config)
    
    trainer = Trainer(config, model, dataset)
    trainer.run_train('testtesttest')

if __name__ == '__main__':
    import configparser
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    trainer_test(config)
    # trainer_finetune_test(config)