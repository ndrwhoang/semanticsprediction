import wandb
import torch

from transformers import RobertaTokenizerFast
from src.dataset.seq2seq_dataset import UDSDataset
from src.model.baseline import BaseModel
from src.model.pretrained_roberta import PretrainedModel
from src.trainer.base_trainer import Trainer



class Interface:
    def __init__(self, config):
        self.config = config
    
    def run_trial_training(self, run_name='subset_test_run', notes=None):
        wandb.init(project='uds', 
                   name=run_name, 
                   notes=notes, 
                   config=self.config
                   )
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        train_dataset = UDSDataset(self.config, 'train_subset', tokenizer)
        model = BaseModel(self.config)
        trainer = Trainer(self.config, model, train_dataset)
        trainer.run_train(run_name)
    
    def run_base_training(self, run_name, notes=None):
        wandb.init(project='uds', 
                   name=run_name, 
                   notes=notes, 
                   config=self.config
                   )
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        train_dataset = UDSDataset(self.config, 'train', tokenizer)
        val_dataset = UDSDataset(self.config, 'val', tokenizer)
        model = BaseModel(self.config)
        trainer = Trainer(self.config, model, train_dataset, val_dataset=val_dataset)
        trainer.run_train(run_name)
    
    def run_pretrained_training(self, run_name, notes=None):
        wandb.init(project='uds', 
                   name=run_name, 
                   notes=notes, 
                   config=self.config
                   )
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        train_dataset = UDSDataset(self.config, 'train', tokenizer)
        val_dataset = UDSDataset(self.config, 'val', tokenizer)
        model = PretrainedModel(self.config)
        trainer = Trainer(self.config, model, train_dataset, val_dataset=val_dataset)
        trainer.run_train(run_name)
    
    def run_pretrained_finetune(self, run_name, notes=None):
        wandb.init(project='uds', 
                   name=run_name, 
                   notes=notes, 
                   config=self.config
                   )
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        train_dataset = UDSDataset(self.config, 'train', tokenizer)
        val_dataset = UDSDataset(self.config, 'val', tokenizer)
        model = PretrainedModel(self.config)
        trainer = Trainer(self.config, model, train_dataset, val_dataset=val_dataset, checkpoint=self.config['model_path']['baseline'])
        trainer.run_train(run_name)
    
    def run_lr_finder(self, run_name='lr_finder', notes=None):
        # TODO: lr_finder
        wandb.init(project='uds', 
                   name=run_name, 
                   notes=notes, 
                   config=self.config
                   )
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        train_dataset = UDSDataset(self.config, 'train', tokenizer)
        val_dataset = UDSDataset(self.config, 'val', tokenizer)
        model = BaseModel(self.config)
        trainer = Trainer(self.config, model, train_dataset, val_dataset)
        raise NotImplementedError
        trainer.run_lr_finder()