import os

import timm
import torch
from transformers import AutoModel, AutoConfig


def init_model(model_name, num_classes):

    if model_name == 'coat_tiny':
        model = timm.create_model(model_name, pretrained=True)
        num_features = model.head.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)

    elif model_name == 'coatnet_rmlp_2_rw_224':
        # printe den aktuellen pfad
        #print(os.path.dirname(os.path.abspath(__file__)))
        model_path = "C:/Users/christian.zwiessler/PycharmProjects/timmpy36/src/lightning/model_architectures/coat_model"
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModel.from_config(config)
        num_features = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)

    elif model_name == 'swin_base_patch4_window12_384':
        model = timm.create_model(model_name)
        num_features = model.head.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)


    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model

def print_model(model):
    print(model)