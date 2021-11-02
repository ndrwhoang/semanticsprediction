import math
import torch
import torch.nn as nn

# Shamelessly stolen implementation
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
    
    def forward(self, batch):
        (input_ids, node_ids, node_labels) = batch
        
        out = self.embedding(input_ids) * math.sqrt(input_ids.size(1))
        out = self.pos_embedding(out)
        out = self.encoder(out)
        out = self.node_out(out)
        
        return out

def model_output_test(config):
    print('starts model output test')
    from torch.utils.data import DataLoader
    from src.dataset.seq2seq_dataset import UDSDataset, collate_fn
    from transformers import RobertaTokenizer
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
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
        out = model(sample)
        print(out.size())  

if __name__ == '__main__':
    import os
    import configparser
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    model_output_test(config)
    
        
        
        

    