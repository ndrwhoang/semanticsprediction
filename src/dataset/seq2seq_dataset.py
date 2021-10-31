import configparser
import json
from tqdm import tqdm
import torch
import os

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# from src.tokenizer.base_tokenizer import Tokenizer
from transformers import AutoTokenizer, RobertaTokenizer

# node_prefix = 'What is the value of nodes <sep> '.split()
# edge_prefix = 'What is the value of edge <sep> '.split()


class CustomDataset(Dataset):    
    def __init__(self, config, mode, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.exclude = ['edge']
        data_path = self.get_data_path(mode)
        self.samples = self.make_samples(data_path)
        
        # self.n_sample = len(self.input_ids)
        
        # print(f'Finished processing data, n_sample: {self.n_sample}')
        
    # def filter_label_set(self, label_ids):
        # TODO: filter the labels to a subset of semantics features using dictionary values
       
    def get_data_path(self, mode):
        if mode == 'train':
            data_path = self.config['data_path']['train_raw']
        elif mode == 'val':
            data_path = self.config['data_path']['val_raw']
        elif mode == 'test':
            data_path = self.config['data_path']['test_raw']
        
        assert data_path is not None
        
        return data_path
    
    def _label2v(self, label_dict, out, edge_label):
        if not edge_label:
            exclude = ['edge']
        else:
            exclude = ['arguement_label', 'predicate_label']
        
        for attribute, values in label_dict.items():
            if isinstance(values, list) or isinstance(values, str):
                continue
            if attribute in exclude:
                continue
            for sub_attr, value in values.items():
                out[self.tokenizer.l_token2id[sub_attr]] = value['value']
                
        return out
    
    def _build_sent_label(self, pred_labels, empty_out):
        sent_label = {}
        
        pred_value = pred_labels['head_token'][1][0] + self.config['dataloading']['offset_value']
        pred_idx = pred_labels['head_token'][0]
        pred_l_v = self._label2v(pred_labels['predicate_label'], empty_out, False)
        
        sent_label[(pred_idx, pred_value)] = pred_l_v
        
        for arg_head, arg_labels in pred_labels['arguement_label'].items():
            arg_value = arg_labels['form'][1][0] + self.config['dataloading']['offset_value']
            arg_idx = arg_labels['form'][0]
            arg_l_v = self._label2v(arg_labels, empty_out, False)
            
            sent_label[(arg_idx, arg_value)] = arg_l_v

        return sent_label
        
    def make_samples(self, data_path):
        print(f'Start procesing sample from raw data {data_path}')
        with open(data_path, 'r') as f:
            samples = json.load(f)
        f.close()
                        
        return samples
    
    def convert_sample_to_input(self, samples):
        # TODO: add edge labels
        print('Start processing sample to input ids')
        input_ids = []
        node_ids, node_labels = [], []
        edge_ids, edge_labels = [], []
        
        for i_sample, sample in enumerate(samples):
            if len(sample['word_labels']) == 0:
                continue
                    
            input_id = [0]      # 2 is end token
            node_id, node_label = [], []   
            
            tokens = sample['sample'].split(' ')
            word_labels = {eval(k):v for k,v in sample['word_labels'].items()}
            
            subword_offset = 1
            for i_token, token in enumerate(tokens):
                token_id = self.tokenizer.encode(token, add_special_tokens=False, add_prefix_space=True)
                input_id.extend(token_id)
                
                if len(token_id) > 1 or subword_offset != 1:
                    if (token, i_token) in word_labels:
                        node_id.append([i_token+subword_offset, i_token+len(token_id)+subword_offset])
                        node_label.append(word_labels[(token, i_token)])
                    subword_offset += len(token_id) - 1
                else:
                    if (token, i_token) in word_labels:
                        node_id.append([i_token+subword_offset])
                        node_label.append(word_labels[(token, i_token)])
                    
            input_id.append(2)
            
            input_ids.append(input_id)
            node_ids.append(node_id)
            node_labels.append(node_label)
            
        return input_ids, node_ids, node_labels
                    
                    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, item):
        return self.input_ids[item], self.node_labels[item], self.edge_labels[item]


if __name__ == '__main__':
    config_path = os.path.join('configs', 'config.cfg')
    config = configparser.ConfigParser()
    config.read(config_path)
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # a = tokenizer.encode('From the AP comes this story :', add_special_tokens=False, add_prefix_space=True)
    
    input_id = [0]      # 2 is end token
    node_id = []
    node_label = []
    word_labels = {
        ('AP', 2): [33], 
        ('comes', 3): [44], 
        ('story', 5): [55]
    }
    edge_labels = {
        (('comes', 3), ('AP', 2)): [55],
        (('comes', 3), ('story', 5)): [66]
    }
    edge_id = []
    edge_ls = []
    for edge, attributes in edge_labels.items():
        edge_tuples = edge
        for edge_tup in edge_tuples:
            edge_ls.append(edge_tup)
    print(edge_ls)
    
    sentence = 'From the AP comes this story :'
    tokens = sentence.split(' ')
    subword_offset = 1
    for i_token, token in enumerate(tokens):
        token_id = tokenizer.encode(token, add_special_tokens=False, add_prefix_space=True)
        input_id.extend(token_id)
        
        if len(token_id) > 1 or subword_offset != 1:
            if (token, i_token) in edge_ls:
                edge_id.append([i_token+subword_offset, i_token+len(token_id)+subword_offset])
            if (token, i_token) in word_labels:
                node_id.append([i_token+subword_offset, i_token+len(token_id)+subword_offset])
                node_label.append(word_labels[(token, i_token)])
            subword_offset += len(token_id) - 1
        else:
            if (token, i_token) in edge_ls:
                edge_id.append([i_token+subword_offset])
            if (token, i_token) in word_labels:
                node_id.append([i_token+subword_offset])
                node_label.append(word_labels[(token, i_token)])
    
    print(edge_id)
    # print(input_id)
    # for i in node_id:
    #     print('=========')
    #     print(i)
    #     if len(i) == 1:
    #         print(tokenizer.decode(input_id[i[0]]))
    #     else:
    #         print(tokenizer.decode(input_id[i[0]: i[-1]]))
    
    # dataset = CustomDataset(config, 'train', tokenizer)
    # print(dataset.input_ids[3])
    # print(dataset.label_ids[3])