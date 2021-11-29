import json
import os
from src.utils import _get_mask
import configparser
from pprint import pprint

def _check_subspace_exist(labels, subspace_idx):
    for label in labels:
        finetuning_nodes = [label[idx] for idx in subspace_idx]
        if any(value is not None for value in finetuning_nodes):
            return True
    
    return False
                

if __name__ == '__main__':
    config_path = os.path.join('configs', 'config.cfg')
    config = configparser.ConfigParser()
    config.read(config_path)
    
    path = 'data\\raw\\train_seq2seq.json'
    
    node_idx, edge_idx = _get_mask(config)
    print(node_idx)
    print(edge_idx)
    
    data_out = {}
    data_path = os.path.join(*path.split('\\'))
    with open(data_path, 'r') as f:
        samples = json.load(f)
        
    for i_sample, (sample_id, sample) in enumerate(samples.items()):
        # if i_sample == 10: break
        added = False
        
        if len(sample['word_labels']) == 0:
            continue
        
        for label in sample['word_labels'].values():
            finetuning_nodes = [label[idx] for idx in node_idx]
            if any(value is not None for value in finetuning_nodes):
                data_out[sample_id] = sample
                added = True
                break
        
        if added == False:
            for label in sample['edge_labels'].values():
                finetuning_nodes = [label[idx] for idx in edge_idx]
                if any(value is not None for value in finetuning_nodes):
                    data_out[sample_id] = sample
        
    with open('data\processed\\train_protoroles_seq2seq.json', 'w') as f:
        json.dump(data_out, f)
        
    
    
        