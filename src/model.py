import os

import timm
import torch
from transformers import AutoModel, AutoConfig


def init_model(model_name, num_classes):

    if model_name == 'coat_tiny':
        model = timm.create_model(model_name, pretrained=True)
        num_features = model.head.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)


    elif model_name == 'coatnet_rmlp_3_rw_224':
        model = timm.create_model(model_name, pretrained=True)
        num_features = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)

    elif model_name == 'coatnet_0_rw_224':
        model = timm.create_model(model_name, pretrained=True)
        num_features = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)

    elif model_name == 'swin_base_patch4_window12_384':
        model = timm.create_model(model_name, pretrained=False)
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src/lightning/model_architectures/swin_base_patch4_window12_384_22kto1k.pth')
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        num_features = model.head.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)

    else:
        raise ValueError(f"CZ: Model {model_name} not supported.")

    return model

def print_model(model):
    print(model)