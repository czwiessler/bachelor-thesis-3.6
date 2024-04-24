import timm
import torch

def init_model(model_name, num_classes):

    if model_name == 'coatnet_0_rw_224.sw_in1k':
        model = timm.create_model(model_name, pretrained=True)
        num_features = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)

    elif model_name == 'coatnet_rmlp_2_rw_224':
        model = timm.create_model(model_name, pretrained=True)
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
        model = timm.create_model(model_name, pretrained=True)
        num_features = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model

def print_model(model):
    print(model)