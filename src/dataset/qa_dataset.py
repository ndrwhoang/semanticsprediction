import configparser
import json
from tqdm import tqdm
import torch
import os

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.tokenizer.base_tokenizer import Tokenizer

# node_prefix = 'What is the value of nodes <sep> '.split()
# edge_prefix = 'What is the value of edge <sep> '.split()


class CustomDataset(Dataset):    
    def __init__(self, config, mode, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.exclude = ['edge']
        data_path = self.get_data_path(mode)
        self.samples = self.make_node_samples(data_path)
        self.input_ids, self.label_ids = self.convert_sample_to_input(self.samples)
        
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
        
    def make_node_samples(self, data_path):
        print(f'Start procesing sample from raw data {data_path}')
        with open(data_path, 'r') as f:
            data = json.load(f)
        f.close()
        
        samples = []
        
        for i_sent, (sent_id, sent) in enumerate(tqdm(data.items())):
            for pred_head, pred_labels in sent.items():
                # empty_out = torch.zeros(len(self.tokenizer.l_token2id))
                empty_out = [0]*len(self.tokenizer.l_token2id)
                sent_label = self._build_sent_label(pred_labels, empty_out)
                
                if len(pred_labels['sample']) == 0:
                    continue
                
                word_seq = pred_labels['sample'].split()
                
                if len(word_seq) > 256:
                    word_seq = word_seq[:256]
                
                label_seq = []
                for i_w, word in enumerate(word_seq):
                    word_label = sent_label.get((i_w, word), empty_out)
                    label_seq.append(word_label)   
                
                # Insert predicate head token indicator
                word_seq.insert(pred_labels['head_token'][0], '<predicate>')
                label_seq.insert(pred_labels['head_token'][0], empty_out)
                
                assert len(word_seq) == len(label_seq)
                sample = {'text': word_seq, 'label': label_seq}
                samples.append(sample)
                        
        return samples
    
    def convert_sample_to_input(self, samples):
        print('Start processing sample to input ids')
        input_ids, label_ids = [], []
        for i_sample, sample in enumerate(tqdm(samples)):
            input_id = [self.tokenizer.s_token2id.get(word, 1) for word in sample['text']]
            assert len(input_id) == len(sample['label'])
            input_ids.append(torch.tensor(input_id, dtype=torch.long))
            label_seq = [torch.tensor(label) for label in sample['label']]
            label_ids.append(torch.stack(label_seq, 0))

        assert len(input_ids) == len(label_ids)
        
        return input_ids, label_ids
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, item):
        return self.input_ids[item], self.label_ids[item]

def collate_fn(batch):
    input_ids, label_ids = zip(*batch)
    
    label_padding = torch.zeros(label_ids[0].size(1), dtype=torch.float)
    input_ids = pad_sequence(input_ids, batch_first=True)
    label_ids = pad_sequence(label_ids, batch_first=True, padding_value=label_padding)
    
    input_ids = torch.stack(input_ids, dim=0)
    label_ids = torch.stack(label_ids, dim=0)
    
    return input_ids, label_ids

if __name__ == '__main__':
    config_path = os.path.join('configs', 'config.cfg')
    config = configparser.ConfigParser()
    config.read(config_path)
    
    tokenizer = Tokenizer(config)
    tokenizer.build_token2id_dict()
    
    dataset = CustomDataset(config, 'train', tokenizer)
    # print(dataset.input_ids[3])
    # print(dataset.label_ids[3])