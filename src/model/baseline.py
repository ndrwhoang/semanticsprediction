import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config['model']  

        self.word_embedding = nn.Embedding(int(self.config['vocab_size']),
                                           int(self.config['embedding_dim']))
        
        self.transformer_layer = nn.TransformerEncoderLayer(int(self.config['d_model']),
                                                            int(self.config['n_head']),
                                                            int(self.config['dim_feedforward']),
                                                            float(self.config['dropout']))
        self.encoder = nn.TransformerEncoder(self.transformer_layer,
                                             int(self.config['n_layer']))
        

    