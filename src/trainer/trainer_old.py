import random
import numpy as np
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.dataset.seq2seq_dataset import collate_fn


class Trainer:
    def __init__(self, config, model, train_dataset, val_dataset=None, test_dataset=None):
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
    
    def set_seed(self):
        self.seed = int(self.config['general']['seed'])
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def _get_optimizer(self, config):
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
        lr_scheduler = None
        
        return optimizer, lr_scheduler
    
    def _get_dataloader(self, config):
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=int(config['training']['bsz_train']),
                                      collate_fn=collate_fn,
                                      shuffle=True,
                                      drop_last=True)
        val_dataloader = DataLoader(self.val_dataset,
                                    batch_size=int(config['training']['bsz_val']),
                                    collate_fn=collate_fn,
                                    shuffle=True,
                                    drop_last=True)
        
        return train_dataloader, val_dataloader
    
    def _index_out_logits(self, raw_logits, node_ids):
        out = []
        for i_sample, sample_ids in enumerate(node_ids):
            sample_out = [raw_logits[i_sample, sample_id] for sample_id in sample_ids]
            sample_out = [torch.mean(word_vec, dim=0) for word_vec in sample_out]
            sample_out = torch.stack(sample_out, 0)
            out.append(sample_out)
        out = pad_sequence(out, batch_first=True, padding_value=10)
        
        return out
    
    def _to_device(self, batch):
        out = []
        for item in batch:
            try:
                out.append(item.to(self.device))
            except:
                out.append(item)
        out = tuple(out)
        
        return out
    
    def run_train(self, run_name: str):
        total_train_step = 0
        
        for epoch in range(int(self.config['training']['n_epoch'])):
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), mininterval=2)
            self.model.train()
            best_loss = float('inf')
            
            for i, batch in pbar:
                # if i == 5: break
                batch = self._to_device(batch)
                (_, node_ids, node_labels, node_masks) = batch
                
                # forward
                raw_logits = self.model(batch)
                output = self._index_out_logits(raw_logits, node_ids)
                                
                loss = self.model.loss_fn(output, node_labels)
                loss = (loss * node_masks.float()).sum()
                loss = loss / node_masks.sum()
                
                # step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.lr_scheduler.step()
                
                # log
                total_train_step += 1
                pbar.set_description(f'(Training) Epoch: {epoch} - Steps: {i}/{len(self.train_dataloader)} - Loss: {loss}', refresh=True)

            val_loss = self.run_validation()
            if val_loss < best_loss:
                best_loss = val_loss
                self._save_model(self.model, self.config['model_path']['base_model'] + run_name)
            
    def run_validation(self):
        pbar = tqdm(enumerate(self.val_dataloader), total = len(self.val_dataloader))
        self.model.eval()
        val_loss = 0
        
        for i, batch in pbar:
            batch = self._to_device(batch)
            (_, node_ids, node_labels, node_masks) = batch
            
            # forward
            raw_logits = self.model(batch)
            output = self._index_out_logits(raw_logits, node_ids)
                        
            loss = self.model.loss_fn(output, node_labels)
            loss = (loss * node_masks.float()).sum()
            loss = loss / node_masks.sum()
            
            val_loss += loss

            pbar.set_description(f'(Validating) Steps: {i}/{len(self.val_dataloader)} - Loss: {loss}', refresh=True)
        
        print(f'Validation loss: {val_loss}')
        
        return val_loss
    
    def _save_model(self, model, path):
        save_path = os.path.join(*path.split('\\'))
        torch.save(model.state_dict(), open(save_path, 'wb'))
    
def trainer_test(config):
    from transformers import RobertaTokenizer
    
    from src.dataset.seq2seq_dataset import UDSDataset
    from src.model.baseline import BaseModel
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    dataset = UDSDataset(config, 'train', tokenizer)
    model = BaseModel(config)
    
    trainer = Trainer(config, model, dataset)
    trainer.run_train()


if __name__ == '__main__':
    import configparser
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    trainer_test(config)