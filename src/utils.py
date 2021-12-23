import torch
import os
import json
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
import numpy as np

# just so pylance shuts up
debug_mode=True

def _index_node_logits(self, raw_logits, node_ids):
    # Indexing relevant words
    # in case of subwords, takes the mean
    # (moved inside model scripts, kept here for furture)
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
    # (moved inside model scripts, kept here for furture)
    edge_outputs = []
    for i_sample, edge_id_pair in enumerate(edge_ids):
        edge_output = [torch.cat((torch.mean(raw_logits[i_sample, idx[0], :], dim=0), torch.mean(raw_logits[i_sample, idx[1], :], dim=0))) for idx in edge_id_pair]
        edge_output = torch.stack(edge_output, dim=0)
        edge_outputs.append(edge_output)
    edge_outputs = pad_sequence(edge_outputs, batch_first=True)
    
    return edge_outputs

def _get_masking_idx(config):
    # Produce mask to exclude semantic subspaces, 
    # change in config 'training' section
    masked_node_subspace = config['training']['node_subspace'].split(' ')
    masked_edge_subspace = config['training']['edge_subspace'].split(' ')
    if masked_node_subspace == ['none'] and masked_edge_subspace == ['none']:
        return [], []
    subspace_dict_path = os.path.join(*config['data_path']['subspace_dict'].split('\\'))
    with open(subspace_dict_path, 'r') as f:
        subspace2id = json.load(f)
    f.close()
    masked_node_idx = [subspace2id[subspace] for subspace in masked_node_subspace]
    masked_edge_idx = [subspace2id[subspace] for subspace in masked_edge_subspace]
    masked_node_idx = list(chain.from_iterable(masked_node_idx))
    masked_edge_idx = list(chain.from_iterable(masked_edge_idx))
    
    return masked_node_idx, masked_edge_idx  

# def _extract_reverse_masks(node_idx, edge_idx, labels, subspace):
#     mask = torch.zeros(labels.size())
#     if subspace == 'nodes':
#         mask[:, :, node_idx] = 1.
#     elif subspace == 'edges':
#         mask[:, :, edge_idx] = 1.
#     else:
#         print('error in choosing subspace to mask')
    
#     return mask

# def _extract_masks(self, labels, subspace):
#     # mask = labels != float('nan')
#     mask = torch.ones(labels.size())
#     if subspace == 'nodes':
#         mask[:, :, self.masked_node_idx] = 0.
#     elif subspace == 'edges':
#         mask[:, :, self.masked_edge_idx] = 0.
#     else:
#         print('error in choosing subspace to mask')
#     mask = mask.to(self.device)
    
#     return mask

def _extract_masks(labels, masked_idx, subspace):
    # mask = labels != float('nan')
    mask = torch.full(labels.size(), np.nan)
    if subspace == 'nodes':
        mask[:, masked_idx] = 1.
    elif subspace == 'edges':
        mask[:, masked_idx] = 1.
    else:
        print('error in choosing subspace to mask')
            
    return mask

def print_debug(input_):
    if debug_mode:
        print(input_)