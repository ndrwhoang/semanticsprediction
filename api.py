from typing import List
from fastapi import FastAPI
from pydantic.parse import load_file
import uvicorn
from pydantic import BaseModel
import torch
import os
import configparser

from src.model.pretrained_roberta import PretrainedModel

class Sample(BaseModel):
    input_ids: torch.Tensor
    node_ids: List
    node_labels: torch.Tensor
    edge_ids: List
    edge_labels: torch.Tensor
    
    class Config:
        arbitrary_types_allowed = True

app = FastAPI(title='semantic prediction api',
              description='semantic prediction api',
              version='1.0')

## Load Model
def load_model():
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    model = PretrainedModel(config)
    checkpoint_path = None
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(os.path.join(*checkpoint_path.split('\\'))))
    
    return model
##

@app.get('/')
def index():
    return {'message': 'Homepage'}

@app.post('/apiv3/')
def api3(sample:Sample):
    return {'message': sample}

@app.post('/prediction')
def run_prediction(sample: Sample):
    sample = sample.dict()
    model = load_model()
    node_pred, edge_pred = model(sample, return_type='output')    
    
    return {'node_pred': node_pred, 'edge_pred': edge_pred}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
    