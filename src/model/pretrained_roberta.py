import logging
import torch
import torch.nn as nn
from transformers import RobertaModel
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PretrainedModel(nn.Module):
    def __init__(self, config):
        # Base mode of 2 transformer encoder layers + 2 linear prediction heads
        # Output is Tanh*5
        super(PretrainedModel, self).__init__()
        self.config = config['model']  

        self.pretrained_encoder = RobertaModel.from_pretrained(self.config['roberta_version'])
        self.node_out = nn.Linear(int(self.config['d_model']), int(self.config['n_node']))
        self.edge_out = nn.Linear(int(self.config['d_model'])*2, int(self.config['n_edge']))
        self.activation = nn.Tanh()
        self.frozen = False
        
        if self.config['freeze_pretrained'] == 'True':
            logger.info('Freezing encoder')
            self.frozen = True
            for param in self.pretrained_encoder.parameters():
                param.requires_grad = False

        self.loss_fn = nn.MSELoss(reduction='none')
    
    def reset_prediciton_heads(self):
        self.node_out.reset_parameters()
        self.edge_out.reset_parameters()
        
    def forward(self, batch, return_type=None):
        if return_type not in ['output', 'loss']:
            raise ValueError('Specify `output` or `loss` return_type')
        # (input_ids, node_ids, _, edge_ids, _) = batch
        input_ids = batch['input_ids']
        node_ids = batch['node_ids']
        node_labels = batch['node_labels']
        edge_ids = batch['edge_ids']
        edge_labels = batch['edge_labels']
        
        out, _ = self.pretrained_encoder(input_ids, return_dict=False)
        
        node_out = self._index_node_logits(out, node_ids)
        edge_out = self._index_edge_logits(out, edge_ids)
        
        node_logits = self.node_out(node_out)
        edge_logits = self.edge_out(edge_out)
        
        node_output = self.activation(node_logits)*5
        edge_output = self.activation(edge_logits)*5
        
        # assert torch.isnan(node_out).any() is not True
        # assert torch.isnan(edge_out).any() is not True
        # assert torch.isnan(node_logits).any() is not True
        # assert torch.isnan(edge_logits).any() is not True
        # assert torch.isnan(node_output).any() is not True
        # assert torch.isnan(edge_output).any() is not True
        
        if return_type == 'output':
            return node_output, edge_output
        elif return_type == 'loss': 
            node_loss = self.loss_fn(node_output, node_labels)
            edge_loss = self.loss_fn(edge_output, edge_labels)
            
            return node_loss, edge_loss  
    
    def _index_node_logits(self, raw_logits, node_ids):
        # Indexing relevant words
        # in case of subwords, takes the mean
        node_outputs = []
        for i_sample, node_id in enumerate(node_ids):
            node_output = [torch.mean(raw_logits[i_sample, idx, :], dim=0) for idx in node_id]
            node_output = torch.stack(node_output, dim=0)
            node_outputs.append(node_output)              
        node_outputs = pad_sequence(node_outputs, batch_first=True)
        
        return node_outputs
    
    def _index_edge_logits(self, raw_logits, edge_ids):
        # Index relevant word pairs and concat
        # in case of subwords, takes the mean
        edge_outputs = []
        for i_sample, edge_id_pair in enumerate(edge_ids):
            edge_output = [torch.cat((torch.mean(raw_logits[i_sample, idx[0], :], dim=0), torch.mean(raw_logits[i_sample, idx[1], :], dim=0))) for idx in edge_id_pair]
            edge_output = torch.stack(edge_output, dim=0)
            edge_outputs.append(edge_output)
        edge_outputs = pad_sequence(edge_outputs, batch_first=True)
        
        return edge_outputs

def model_output_test(config):
    # Test function
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
    model = PretrainedModel(config)
    model.reset_prediciton_heads()
    
    # for i, sample in enumerate(dataloader):
    #     if i == 3: break  
    #     print('==========')
    #     node_logits, edge_logits = model(sample)
    #     print(node_logits.size())  
    #     print(edge_logits.size())

if __name__ == '__main__':
    import os
    import configparser
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    model_output_test(config)
    
        
        
        

    