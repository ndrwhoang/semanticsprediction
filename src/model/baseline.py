import math
import torch
import torch.nn as nn
from typing import Dict, List
from torch.nn.utils.rnn import pad_sequence

# Shamelessly stolen implementation
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x += self.pe[:x.size(0), :]
        return self.dropout(x)

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config['model']  

        self.pos_embedding = PositionalEncoding(int(self.config['d_embedding']), 
                                                0.1, 
                                                int(self.config['max_len']))
        self.embedding = nn.Embedding(int(self.config['vocab_size']),
                                      int(self.config['d_embedding']))
        
        self.transformer_layer = nn.TransformerEncoderLayer(int(self.config['d_model']),
                                                            int(self.config['n_head']),
                                                            int(self.config['dim_feedforward']),
                                                            float(self.config['dropout']))
        self.encoder = nn.TransformerEncoder(self.transformer_layer,
                                             int(self.config['n_layer']))
        self.node_out = nn.Linear(int(self.config['d_model']), int(self.config['n_node']))
        self.edge_out = nn.Linear(int(self.config['d_model'])*2, int(self.config['n_edge']))
        self.activation = nn.Tanh()

        self.loss_fn = nn.MSELoss(reduction='none')
        
    def forward(self, batch):
        (input_ids, node_ids, node_labels, edge_ids, edge_labels) = batch
        
        out = self.embedding(input_ids) * math.sqrt(input_ids.size(1))
        out = self.pos_embedding(out)
        out = self.encoder(out)
        
        node_out = self._index_node_logits(out, node_ids)
        edge_out = self._index_edge_logits(out, edge_ids)
        
        node_logits = self.node_out(node_out)
        edge_logits = self.edge_out(edge_out)
        
        node_output = self.activation(node_logits)
        edge_output = self.activation(edge_logits)
        
        return node_output, edge_output
    
    def _index_node_logits(self, raw_logits, node_ids):
        node_outputs = []
        for i_sample, node_id in enumerate(node_ids):
            node_output = [torch.mean(raw_logits[i_sample, idx, :], dim=0) for idx in node_id]
            node_output = torch.stack(node_output, dim=0)
            node_outputs.append(node_output)              
        node_outputs = pad_sequence(node_outputs, batch_first=True)
        
        return node_outputs
    
    def _index_edge_logits(self, raw_logits, edge_labels):
        edge_outputs = []
        for i_sample, edge_id_pair in enumerate(edge_labels):
            edge_output = [torch.cat((torch.mean(raw_logits[i_sample, idx[0], :], dim=0), torch.mean(raw_logits[i_sample, idx[1], :], dim=0))) for idx in edge_id_pair]
            edge_output = torch.stack(edge_output, dim=0)
            edge_outputs.append(edge_output)
        edge_outputs = pad_sequence(edge_outputs, batch_first=True)
        
        return edge_outputs

def model_output_test(config):
    print('starts model output test')
    from torch.utils.data import DataLoader
    from src.dataset.seq2seq_dataset import UDSDataset, collate_fn
    from transformers import RobertaTokenizerFast
    
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    dataset = UDSDataset(config, 'train', tokenizer)
    dataloader = DataLoader(dataset, 
                            batch_size=4, 
                            shuffle=True,
                            drop_last=True,
                            collate_fn=collate_fn)
    model = BaseModel(config)
    
    for i, sample in enumerate(dataloader):
        if i == 3: break  
        print('==========')
        node_logits, edge_logits = model(sample)
        print(node_logits.size())  
        print(edge_logits.size())

if __name__ == '__main__':
    import os
    import configparser
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    model_output_test(config)
    
        
        
        

    