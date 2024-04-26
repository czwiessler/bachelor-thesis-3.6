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

        #model = timm.create_model(model_name, pretrained=False)
        num_features = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)

    elif model_name == 'resnet18':
        model = timm.create_model(model_name, pretrained=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)


    elif model_name == 'timm/maxvit_xlarge_tf_512.in21k_ft_in1k':
        model = timm.create_model(model_name, pretrained=True)
        num_features = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)

    elif model_name == 'timm/maxvit_xlarge_tf_384.in21k_ft_in1k':
        model = timm.create_model(model_name, pretrained=True)
        num_features = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)

    elif model_name == 'timm/maxvit_base_tf_384.in21k_ft_in1k':
        model = timm.create_model(model_name, pretrained=True)
        num_features = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)

    elif model_name == 'timm/davit_base.msft_in1k':
        model = timm.create_model(model_name, pretrained=False)
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lightning', 'model_architectures')
        model.load_state_dict(torch.load(data_path + '/model_davit_base_weights.bin'))
        num_features = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model

def print_model(model):
    print(model)