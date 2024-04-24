import json
import time
import torch
from pytorch_lightning import LightningModule
import torchvision.transforms as transforms
from PIL import Image
import os
import src.model as model

model_name = 'timm/davit_base.msft_in1k'

# choose from the following models:
# coatnet_0_rw_224.sw_in1k
# resnet18
# google/vit-base-patch16-224
# google/vit-base-patch32-384
# timm/maxvit_xlarge_tf_512.in21k_ft_in1k
# timm/maxvit_xlarge_tf_384.in21k_ft_in1k
# timm/maxvit_base_tf_384.in21k_ft_in1k
# timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k
# timm/davit_base.msft_in1k
# microsoft/swinv2-base-patch4-window12-192-22k

class CustomModel(LightningModule):
    def __init__(self, model_name, num_classes=8, checkpoint_path=None, image_size=224):
        super().__init__()
        # Initialisiere das gewählte Modell
        self.model = model.init_model(model_name=model_name, num_classes=num_classes)
        self.image_size = image_size

        # Wenn ein Checkpoint-Pfad bereitgestellt wird, lade den Checkpoint
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint['state_dict'])

    def forward(self, x):
        return self.model(x)

#
def create_transformer(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.409762978553772, 0.3795592486858368, 0.3669155538082123],
                             std=[0.22302936017513275, 0.21662043035030365, 0.23332579433918])
    ])

# Stellen Sie sicher, dass predict_image die custom_model.eval() aufruft
def predict_image(image_path, model):
    model.eval()
    image = Image.open(image_path)

    transform = create_transformer(model.image_size)

    image = transform(image).unsqueeze(0)  # adding a batch dim

    start_time = time.time()
    with torch.no_grad():
        output = model(image)
        if model_name == 'google/vit-base-patch16-224' or model_name == 'google/vit-base-patch32-384':
            output = output.logits
        probabilities = torch.softmax(output, dim=1)  # Conversion of logits into probabilities
        top_probabilities, predicted = torch.topk(probabilities, 3)  # Top-2 Vorhersagen und ihre Wahrscheinlichkeiten

    prediction_time = time.time() - start_time

    return predicted, top_probabilities, prediction_time


def load_class_names(json_file_path):
    with open(json_file_path, 'r') as file:
        class_names = json.load(file)
    return class_names


def extract_class_from_filename(filename):
    """Extrahiert die Klasse aus dem Dateinamen."""
    parts = filename.split('_')
    return '_'.join(parts[:2])


def find_class_id(class_name, class_names):
    """Findet die Klassen-ID basierend auf dem Klassennamen."""
    for key, value in class_names.items():
        if value == class_name:
            return int(key)
    return -1


def predict_and_format(image_path, model, class_names):
    prediction = predict_image(image_path, model)
    predicted_indices = prediction[0][0]
    probabilities = prediction[1][0]
    prediction_time = prediction[2]
    return predicted_indices, probabilities, prediction_time


def print_header():
    header = "| {:<40} | {:<15} | {:<10} | {:<40} | {:<15} |".format("Image", "Result", "Class-ID", "Top-k IDs",
                                                                     "Time (seconds)")
    print(header)
    print("-" * len(header))
    return header


def print_prediction(image_name, is_correct, class_id, top_k_ids, top_k_probs, prediction_time):
    result_text = "correct" if is_correct else "not correct"
    top_k_text = ", ".join(f"{id} ({prob:.2f}%)" for id, prob in zip(top_k_ids, top_k_probs))
    print("| {:<40} | {:<15} | {:<10} | {:<40} | {:<15.5f} |".format(image_name, result_text, class_id, top_k_text,
                                                                     prediction_time))


def main(model_name):
    # Konfiguration für das gewünschte Modell und Checkpoint
    with open('eval_configs.json', 'r') as config_file:
        configs = json.load(config_file)
    model_name_to_load = model_name
    selected_config = next((config for config in configs if config['model_name'] == model_name_to_load), None)

    if selected_config:
        custom_model = CustomModel(**selected_config)
        custom_model.eval()
    else:
        print(f"Kein Modell mit dem Namen {model_name_to_load} gefunden.")

    json_file_path = '../../classes.json'
    class_names = load_class_names(json_file_path)
    image_folder_path = '../../../data/test'
    k = 5
    total_time = 0
    count = 0
    correct_count = 0
    total_top1_confidence = 0

    header = print_header()

    for image_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, image_name)
        actual_class_name = extract_class_from_filename(image_name)
        actual_class_id = find_class_id(actual_class_name, class_names)

        predicted_indices, probabilities, prediction_time = predict_and_format(image_path, custom_model, class_names)
        # only the first prediction is considered        is_correct = actual_class_id in predicted_indices[:k].tolist()
        is_correct = actual_class_id == predicted_indices[0].item()
        # Erhalte die Top-k IDs und deren Wahrscheinlichkeiten
        top_k_indices = predicted_indices[:k].tolist()
        top_k_probs = [prob * 100 for prob in probabilities[:k].tolist()]
        total_top1_confidence += top_k_probs[0]

        print_prediction(image_name, is_correct, actual_class_id, top_k_indices, top_k_probs, prediction_time)

        total_time += prediction_time
        count += 1
        correct_count += is_correct

    average_time = total_time / count if count else 0
    average_top1_confidence = total_top1_confidence / count if count else 0

    print("-" * len(header))
    print(f"Average prediction time per image: {average_time:.5f} seconds")
    print(f"Average Top-1 confidence: {average_top1_confidence:.2f}%")
    print(f"Accuracy: {correct_count}/{count} correct ({correct_count / count * 100:.2f}%)")


if __name__ == "__main__":
    main(model_name=model_name)
