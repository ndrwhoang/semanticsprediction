import re
import json
from torch._C import dtype
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List

# node_prefix = 'What is the value of nodes <sep> '.split()
# edge_prefix = 'What is the value of edge <sep> '.split()


class UDSDataset(Dataset):    
    def __init__(self, config, mode, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.exclude = ['edge']
        data_path = self.get_data_path(mode)
        self.samples = self.make_samples(data_path)
        self.input_ids, self.node_ids, self.node_labels, self.edge_ids, self.edge_labels = self.convert_sample_to_input(self.samples)
        
        self.n_sample = len(self.input_ids)
        print(f'Finished processing data, n_sample: {self.n_sample}')
        
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
    
    def make_samples(self, data_path):
        print(f'Start procesing sample from raw data {data_path}')
        with open(data_path, 'r') as f:
            samples = json.load(f)
        f.close()
                        
        return samples
    
    def convert_sample_to_input(self, samples):
        print('Start processing sample to input ids')
        input_ids = []
        node_ids, node_labels = [], []
        edge_ids, edge_labels = [], []
        
        for i_sample, (sample_id, sample) in enumerate(samples.items()):
            if len(sample['word_labels']) == 0:
                continue
            
            # Add input_id (List[int]): word ids
            tokenizer_out = self.tokenizer(sample['sample'], 
                                      add_special_tokens=False, 
                                      return_offsets_mapping=True)
            input_id = tokenizer_out['input_ids']
            input_ids.append(input_id)
            
            
            offset_mapping = tokenizer_out['offset_mapping'] 
            original_indices = [(m.start(), m.end()) for m in re.finditer(r'\S+', sample['sample'])]    # https://stackoverflow.com/questions/13734451/string-split-with-indices-in-python
            
            # Add node_label (Dict{(tuple):List(int)}): dictionary of subwords to attribute value
            node_label = [torch.tensor(np.array(v, dtype=float), dtype=torch.float) for v in sample['word_labels'].values()] 
            node_id = []
            for word in sample['word_labels'].keys():
                alignment = self._find_alignment_for_token(eval(word), offset_mapping, original_indices)
                node_id.append(alignment)
            node_ids.append(node_id)
            node_labels.append(torch.stack(node_label, dim=0))
            
            # Add edge_label (Dict{tuple(tuple(int), tuple(int)): List(int)})
            placeholder_tensor = np.empty((int(self.config['model']['n_edge']),))
            placeholder_tensor[:] = np.nan
            edge_id = []
            if len(sample['edge_labels']) == 0:
                edge_label = [torch.tensor(placeholder_tensor, dtype=torch.float)]
                edge_id.append(((0,), (0,)))
            else:
                edge_label = [torch.tensor(np.array(v, dtype=float), dtype=torch.float) for v in sample['edge_labels'].values()]
                for word_pair in sample['edge_labels'].keys():
                    first_alignment = self._find_alignment_for_token(eval(word_pair)[0], offset_mapping, original_indices)
                    sec_alignment = self._find_alignment_for_token(eval(word_pair)[1], offset_mapping, original_indices)
                    edge_id.append((first_alignment, sec_alignment))
            edge_ids.append(edge_id)
            edge_labels.append(torch.stack(edge_label, dim=0))
            
            assert len(input_ids) == len(node_ids)
            assert len(input_ids) == len(node_labels)
            assert len(input_ids) == len(edge_ids)
            assert len(input_ids) == len(edge_labels)
            
            
        return input_ids, node_ids, node_labels, edge_ids, edge_labels
        
    def _find_alignment_for_token(self, token, offset_mapping, original_indices):
        label_id = original_indices[token[-1]]
        try:
            aligned = (offset_mapping.index(label_id),)
        except ValueError:
            for tok_indices in offset_mapping:
                if tok_indices[0] == label_id[0]:
                    first_subword_id = offset_mapping.index(tok_indices)
                    for tok_indices_rest in offset_mapping[first_subword_id:]:
                        if tok_indices_rest[1] == label_id[1]:
                            last_subword_id = offset_mapping.index(tok_indices_rest)
                            break
                    break
            try:
                aligned = (first_subword_id, last_subword_id)
            except UnboundLocalError:
                # Shouldn't run into this unless a subtoken is longer than token
                print(original_indices)
                print(offset_mapping)
                print(label_id)
        
        return aligned
                  
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, item):
        return self.input_ids[item], self.node_ids[item], self.node_labels[item], self.edge_ids[item], self.edge_labels[item]

def collate_fn(batch):
    input_ids, node_ids, node_labels, edge_ids, edge_labels = zip(*batch)
    
    input_ids = [torch.tensor(input_id) for input_id in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True)    
    node_labels = pad_sequence(node_labels, batch_first=True)
    edge_labels = pad_sequence(edge_labels, batch_first=True)
    
    return input_ids, node_ids, node_labels, edge_ids, edge_labels

def _collate_fn(batch):
    # for old processing pipeline, keep for posterity
    input_ids, node_ids, node_labels = zip(*batch)
    
    input_ids = [torch.tensor(input_id) for input_id in input_ids]
    # node_ids = [torch.tensor(node_id) for node_id in node_ids]
    node_labels = [torch.tensor(node_label) for node_label in node_labels]
    
    label_padding = torch.zeros(len(node_labels[0]), dtype=torch.float)
    input_ids = pad_sequence(input_ids, batch_first=True)
    node_labels = pad_sequence(node_labels, batch_first=True, padding_value=10)
    node_masks = node_labels != 10
    
    
    # for sample_node_label in node_labels:
    #     for node_id, label_vec in sample_node_label.items():
    #         node_mask = [1 if value is not 10 else 0 for value in label_vec]
    #         sample_node_label[node_id] = [value if value is not None else 10 for value in label_vec]
    #         node_masks.append(node_mask)
    #         node_labels_processed.append(sample_node_label[node_id])
    
    
    return input_ids, node_ids, node_labels, node_masks

def dataloader_test(dataset):
    print('dataloader test')
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    for i, batch in enumerate(dataloader):
        # if i == 3: break
        print('===========')
        print('*')
        print('*')
        print('*')
        input_ids, node_ids, node_labels, edge_ids, edge_labels = batch
        
        a = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        print(a)
        print(node_ids[0])
        b = [a[i[0]:i[-1]+1] for i in node_ids[0]]
        print(b)
        # print(node_labels[0].tolist())
        # for label in node_labels[0].tolist():
        #     print(label)
        assert len(b) == node_labels.size(1)
        # print(edge_ids[0])
        # print(edge_labels[0].tolist())
        # print(edge_labels.size())
        
        # print(input_ids.size())
        # print([len(node_id) for node_id in node_ids])
        # print(node_labels.size())
        # print([len(edge_id) for edge_id in edge_ids])
        # print(edge_labels.size())
        
        # print(node_ids)
        # print(edge_ids)
        
        
def index_offset_test_2():
    example_str = 'The sheikh in wheel - chair has been attacked with a F - 16 - launched bomb .'
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    a = tokenizer(example_str, add_special_tokens=False, return_offsets_mapping=True)

    print(a['offset_mapping'])
    # original_indices = [(m.group(0), (m.start(), m.end())) for m in re.finditer(r'\S+', example_str)]
    original_indices = [(m.start(), m.end()) for m in re.finditer(r'\S+', example_str)]
    print(original_indices)    
        
    label_id = original_indices[1]
    print(label_id)
    try:
        aligned = (a['offset_mapping'].index(label_id))
    except ValueError:
        for tok_indices in a['offset_mapping']:
            if tok_indices[0] == label_id[0]:
                first_subword_id = a['offset_mapping'].index(tok_indices)
                for tok_indices_rest in a['offset_mapping'][first_subword_id:]:
                    if tok_indices_rest[1] == label_id[1]:
                        last_subword_id = a['offset_mapping'].index(tok_indices_rest)
                        break
                break
        aligned = (first_subword_id, last_subword_id)
    
    print(aligned)

if __name__ == '__main__':
    import os
    import configparser
    from torch.utils.data import DataLoader
    # from src.tokenizer.base_tokenizer import Tokenizer
    from transformers import RobertaTokenizerFast
        
    config_path = os.path.join('configs', 'config.cfg')
    config = configparser.ConfigParser()
    config.read(config_path)
    
    example_str = 'The sheikh in wheel - chair has been attacked with a F - 16 - launched bomb .'
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    a = tokenizer(example_str, add_special_tokens=False, return_offsets_mapping=True)
    # print(tokenizer.convert_ids_to_tokens(a['input_ids']))
    
    
    dataset = UDSDataset(config, 'train', tokenizer)
    # print(dataset.input_ids[0])
    # print(dataset.node_labels[0])
    # print(dataset.edge_labels[0])
    # mask = (dataset.node_labels[0] == dataset.node_labels[0]).int().float()
    # print(mask)
    # mask[:, [1, 3, 4, 5]] = 0
    # print(mask)
    # print(torch.nan_to_num(dataset.node_labels[0])*mask)
    # print(torch.nan_to_num(dataset.node_labels[0]))
    
    dataloader_test(dataset)
    