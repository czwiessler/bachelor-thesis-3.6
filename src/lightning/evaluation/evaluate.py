import json
import time
import torch
from pytorch_lightning import LightningModule
import torchvision.transforms as transforms
from PIL import Image
import os
import src.model as model
from src.data_preprocessing import get_input_format

model_name = 'timm/davit_base.msft_in1k'

# choose from the following models:
# coat_tiny
# swin_base_patch4_window12_384
# coatnet_0_rw_224
# coatnet_rmlp_3_rw_224
# maxvit_base_224
# timm/davit_base.msft_in1k

class CustomModel(LightningModule):
    def __init__(self, model_name, num_classes=8, checkpoint_path=None, image_size=224):
        super().__init__()
        self.model = model.init_model(model_name=model_name, num_classes=num_classes)
        self.image_size = image_size
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint['state_dict'])

    def forward(self, x):
        return self.model(x)


def create_transformer(size, image_path):
    folder_path = os.path.dirname(os.path.dirname(image_path))

    image_format, ratio = get_input_format(folder_path)

    resized_width = int(size * ratio)
    resized_height = size

    return transforms.Compose([
        transforms.Resize((resized_height, resized_width)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5890, 0.5702, 0.5509],
                             std=[0.2237, 0.2317, 0.2350])
    ])


def predict_image(image_path, model):
    model.eval()
    image = Image.open(image_path)

    transform = create_transformer(model.image_size, image_path)

    image = transform(image).unsqueeze(0)  # adding a batch dim

    start_time = time.time()
    with torch.no_grad():
        output = model(image)
        if model_name == 'google/vit-base-patch16-224' or model_name == 'google/vit-base-patch32-384':
            output = output.logits
        probabilities = torch.softmax(output, dim=1)  # Conversion of logits into probabilities
        top_probabilities, predicted = torch.topk(probabilities, 3)

    prediction_time = time.time() - start_time

    return predicted, top_probabilities, prediction_time

def predict_and_format(image_path, model):
    prediction = predict_image(image_path, model)
    predicted_indices = prediction[0][0]
    probabilities = prediction[1][0]
    prediction_time = prediction[2]
    return predicted_indices, probabilities, prediction_time

def find_class_id(class_folder, image_folder_path):
    base_path = image_folder_path
    # List all directories in the specified base path
    try:
        directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    except FileNotFoundError:
        print(f"Error: The directory {base_path} was not found.")
        return -1  # Return -1 or handle error as appropriate

    # Sort directories to ensure consistent ordering
    directories.sort()

    # Try to find the index of the class_folder
    try:
        class_id = directories.index(class_folder)
    except ValueError:
        print(f"Error: The folder '{class_folder}' was not found in {base_path}.")
        return -1  # Return -1 or handle error as appropriate

    return class_id

def print_header():
    header = "| {:<40} | {:<15} | {:<10} | {:<40} | {:<15} |".format("Image", "Result", "Class-ID", "Top-k IDs",
                                                                     "Time (seconds)")
    print(header)
    print("-" * len(header))
    return header



def print_prediction(image_name, is_correct, class_name, top_k_ids, top_k_probs, prediction_time):
    result_text = "correct" if is_correct else "not correct"
    # Generiere eine formatierte String-Liste von Vorhersagen und Wahrscheinlichkeiten
    top_k_text = ", ".join(f"{id} ({prob:.2f}%)" for id, prob in zip(top_k_ids, top_k_probs))
    print("| {:<40} | {:<15} | {:<10} | {:<40} | {:<15f} |".format(image_name, result_text, class_name, top_k_text, prediction_time))



def main(model_name):
    with open('../model_architectures/model_configs.json', 'r') as config_file:
        configs = json.load(config_file)
    selected_config = next((config for config in configs if config['model_name'] == model_name), None)
    if selected_config:
        custom_model = CustomModel(**selected_config)
        custom_model.eval()
    else:
        print(f"Kein Modell mit dem Namen {model_name} gefunden.")

    image_folder_path = '../../../custom_data/test'
    header = print_header()
    count = correct_count = total_time = total_top1_confidence = 0

    for class_folder in os.listdir(image_folder_path):
        folder_path = os.path.join(image_folder_path, class_folder)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            predicted_indices, top_k_probs, prediction_time = predict_and_format(image_path, custom_model)
            top_k_probs = [prob * 100 for prob in top_k_probs]  # Convert probabilities to percentage format
            is_correct = True if predicted_indices[0] == find_class_id(class_folder, image_folder_path) else False
            top_k_indices = predicted_indices[:3].tolist()  # Adjust the number of predictions if needed
            print_prediction(image_name, is_correct, class_folder, top_k_indices, top_k_probs, prediction_time)
            total_time += prediction_time
            count += 1
            correct_count += is_correct
            total_top1_confidence += top_k_probs[0]

    average_time = total_time / count if count else 0
    average_top1_confidence = total_top1_confidence / count if count else 0
    print("-" * len(header))
    print(f"Average prediction time per image: {average_time:.5f} seconds")
    print(f"Average Top-1 confidence: {average_top1_confidence:.2f}%")
    print(f"Accuracy: {correct_count}/{count} correct ({correct_count / count * 100:.2f}%)")

if __name__ == "__main__":
    main(model_name=model_name)
