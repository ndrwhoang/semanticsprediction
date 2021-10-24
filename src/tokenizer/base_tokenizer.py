import os
import json
import configparser
from tqdm import tqdm

class Tokenizer:
    def __init__(self, config):
        self.config = config
    
    def init_vocab(self):
        # Create dictionaries of token count from the dataset
        data_path = self.config['data_path']['train_raw']
        with open(data_path, 'r') as f:
            data = json.load(f)
        f.close()
        print(f'Start building for vocab from the training dataset {data_path}')
        
        vocab = {}
        label_vocab = {}
        
        for sample_id, preds in tqdm(data.items()):
            # Loop through predicate heads
            for pred_id, pred in preds.items():
                
                # Word dictionary
                text = pred['sample'].split()
                for word in text:
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        vocab[word] += 1
                
                # Label dictionary
                for attribute, values in pred['predicate_label'].items():
                    if isinstance(values, list) or isinstance(values, str):
                        continue
                    for sub_attr, value in values.items():
                        if sub_attr not in label_vocab:
                            label_vocab[sub_attr] = 1
                        else:
                            label_vocab[sub_attr] += 1
                
                for arg_head, arg_labels in pred['arguement_label'].items():
                    for attribute, values in arg_labels.items():
                        if isinstance(values, list) or isinstance(values, str):
                            continue
                        for sub_attr, value in values.items():
                            if sub_attr not in label_vocab:
                                label_vocab[sub_attr] = 1
                            else:
                                label_vocab[sub_attr] += 1
                
                # TODO: add edge labels
                for edge_head, edge_labels in pred['edge'].items():
                    for attribute, values in edge_labels.items():
                        if isinstance(values, list) or isinstance(values, str):
                            continue
                        if attribute not in label_vocab:
                            label_vocab[attribute] = 1
                        else:
                            label_vocab[attribute] += 1
                                
        self.vocab = vocab
        self.label_vocab = label_vocab
        
        print(f'Finished processing, vocab len: {len(self.vocab)} - label len: {len(self.label_vocab)}')
        print(f"Dumping to files {self.config['data_path']['vocab']} and {self.config['data_path']['label_vocab']}")
        
        # Dump dictionaries to be loaded later
        self._dump_json_vocab(self.vocab, self.config['data_path']['vocab'])
        self._dump_json_vocab(self.label_vocab, self.config['data_path']['label_vocab'])
    
    def build_token2id_dict(self):
        # Create token 2 id dictionaries from saved vocab files
        print('Loading token to ids dictionaries from saved vocab files')
        self.vocab, self.label_vocab = self._load_vocab()
        self.vocab = {k: v for k, v in sorted(self.vocab.items(), key=lambda item: item[1])}
        s_token2id, l_token2id = {'<pad>': 0, '<unk>': 1, '<sep>': 2, '<predicate>': 3}, {}
        n_special = len(s_token2id)
        
        # token2id for input string
        for i, key in enumerate(self.vocab.keys()):
            if i == int(self.config['dataloading']['vocab_limit']):
                break
            s_token2id[key] = i + n_special
        
        # token2id for label
        for i, key in enumerate(self.label_vocab.keys()):
            l_token2id[key] = i

        self.s_token2id = s_token2id
        self.l_token2id = l_token2id
        
        self._dump_json_vocab(self.s_token2id, self.config['data_path']['vocab_dict'])
        self._dump_json_vocab(self.l_token2id, self.config['data_path']['label_vocab_dict'])
        
        print('Finished loading')
        print(f'vocab len: {len(self.s_token2id)}')
        print(f'label vocab len: {len(self.l_token2id)}')
        
    def _load_vocab(self):
        try:
            vocab_path = self.config['data_path']['vocab']
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            f.close()
            
            label_vocab_path = self.config['data_path']['label_vocab']
            with open(label_vocab_path, 'r') as f:
                label_vocab = json.load(f)
            f.close()
            
        except FileNotFoundError:
            print(f"Vocab file {self.config['data_path']['vocab']} not found or label vocab file {self.label_vocab['data_path']['label_vocab']} not found")
        
        return vocab, label_vocab
    
    def _dump_json_vocab(self, dict_, path):
        with open(path, 'w') as f:
            json.dump(dict_, f)
        f.close()
                

if __name__ == '__main__':
    config_path = os.path.join('configs', 'config.cfg')
    config = configparser.ConfigParser()
    config.read('configs\config.cfg')
    # print(config.sections())
    
    tokenizer = Tokenizer(config)
    tokenizer.init_vocab()
    tokenizer.build_token2id_dict()
    print(tokenizer.l_token2id)
    
    
                
                