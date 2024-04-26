import json
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mean_and_std(directory: str):
    """ Berechnet den Mittelwert und die Standardabweichung der Bilder im angegebenen Verzeichnis. """

    # Transformation definieren, um Bilder in PyTorch Tensoren zu konvertieren
    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.CenterCrop(192),
        transforms.ToTensor()
    ])

    # Bildordner laden
    dataset = datasets.ImageFolder(directory, transform=transform)

    # DataLoader erstellen
    loader = DataLoader(dataset, batch_size=10, num_workers=1, shuffle=False)

    # Mittelwert und Standardabweichung berechnen
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return mean, std

def calculate_mean_std(root_dir):
    # Liste zum Speichern der Tensoren
    data_list = []

    # Transformation, um PIL-Bilder in Tensoren umzuwandeln und zu normalisieren
    transform = transforms.ToTensor()

    # Durchläuft jedes Bild im Verzeichnis
    for file in os.listdir(root_dir):
        # Unterstützung für gängige Bildformate
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            # Bild laden
            image_path = os.path.join(root_dir, file)
            image = Image.open(image_path).convert('RGB')  # Konvertierung in RGB

            # Bild in einen Tensor umwandeln
            tensor = transform(image)

            # Tensor zur Liste hinzufügen
            data_list.append(tensor)

    # Überprüfung, ob Bilder gefunden wurden
    if not data_list:
        raise ValueError("Keine Bilder im angegebenen Verzeichnis gefunden.")

    # Alle Tensoren entlang einer neuen Dimension stapeln
    stacked_tensors = torch.stack(data_list)

    # Berechnung des Mittelwerts und der Standardabweichung
    mean = stacked_tensors.mean(dim=[0, 2, 3])
    std = stacked_tensors.std(dim=[0, 2, 3])

    return mean.tolist(), std.tolist()

# mean, std = calculate_mean_std('C:/Users/christian.zwiessler/Documents/Uni/datasets/friend or foe/transparent_objects/transparent_objects/Misc/Data/Full/RGB')
# all data old [0.409762978553772, 0.3795592486858368, 0.3669155538082123] [0.22302936017513275, 0.21662043035030365, 0.23332579433918]
# all data (tensor([0.5890, 0.5702, 0.5509]), tensor([0.2237, 0.2317, 0.2350]))
# train data: ([0.4117877781391144, 0.3814772963523865, 0.36748409271240234], [0.22235986590385437, 0.21537397801876068, 0.2318010926246643])

def debug_transforms():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'train')

    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.CenterCrop(192),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5890, 0.5702, 0.5509],
                             std=[0.2237, 0.2317, 0.2350]),
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for images, _ in loader:
        print("Min:", images.min().item(), "Max:", images.max().item())
        break  # Remove this to see values for more images


def get_input_format(data_path):
    dataset = datasets.ImageFolder(data_path)
    first_image = dataset[0][0]
    # check if all images of the dataset have the same size
    """for i in range(1, len(dataset)):
        image = dataset[i][0]
        if image.size != first_image.size:
            raise ValueError("Images in the dataset have different sizes.")"""
    width, height  = first_image.size
    ratio = width / height
    return first_image.size, ratio


def create_data_loader(model_name):
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'custom_data', 'train')
    batch_size = 8
    train_split = 0.8

    # Step 1: Prepare the data
    # Pfad zur Konfigurationsdatei
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src/lightning/model_architectures/model_configs.json')

    # Lade Konfigurationsdaten
    with open(config_path, 'r') as file:
        configs = json.load(file)

    # Suche nach der passenden Konfiguration für das gegebene Modell
    model_config = next((config for config in configs if config['model_name'] == model_name), None)

    # Überprüfe, ob eine Konfiguration gefunden wurde
    if model_config is None:
        raise ValueError(f"CZ: Model {model_name} not supported.")

    image_format, ratio = get_input_format(data_path)
    size = model_config['image_size']  # Größe aus der Konfiguration extrahieren

    resized_width = int(size * ratio)
    resized_height = size

    # Transformationspipeline definieren
    transform = transforms.Compose([
        transforms.Resize((resized_height, resized_width)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5890, 0.5702, 0.5509],
                             std=[0.2237, 0.2317, 0.2350]),
    ])

    if model_name == 'microsoft/swinv2-base-patch4-window12-192-22k':
        transform = transforms.Compose([
            *transform.transforms,  # Hier fügen wir die bestehenden Transformationen aus der alten Pipeline ein
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
        ])

    """ 
    not used transformations
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    """

    # Create a single dataset from the folder
    dataset = ImageFolder(root=data_path, transform=transform)
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size

    """
    # Get class distribution and calculate weights for each sample
    class_counts = torch.tensor([t[1] for t in dataset.samples]).bincount()
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[dataset.targets]

    # Setup the samplers
    train_sampler = WeightedRandomSampler(sample_weights, num_samples=train_size, replacement=True)
    val_sampler = WeightedRandomSampler(sample_weights, num_samples=val_size, replacement=True)
    """

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    # Step 2: Setup DataLoader
    """
    # Setup DataLoader using the samplers
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    """

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == '__main__':
    #print(get_mean_and_std("C:/Users/christian.zwiessler/PycharmProjects/bachelor-thesis/custom_data/full_data"))
    # (tensor([0.5890, 0.5702, 0.5509]), tensor([0.2237, 0.2317, 0.2350]))
    print(get_input_format("C:/Users/christian.zwiessler/PycharmProjects/bachelor-thesis/custom_data/full_data"))