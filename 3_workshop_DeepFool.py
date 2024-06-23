import os
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image

from art.attacks.evasion import DeepFool
from art.estimators.classification import PyTorchClassifier

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.convnext.feature_extraction_convnext")

# Load the pre-trained model and feature extractor from Hugging Face
model_name = "microsoft/resnet-50"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Wrapper class for the model
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits

# ART requires a classifier wrapper around the PyTorch model
def create_art_classifier(model):
    wrapped_model = ModelWrapper(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0, 1),
        preprocessing=(0, 1)
    )
    return classifier

art_classifier = create_art_classifier(model)

# Function to limit perturbation
def limit_perturbation(original, perturbed, max_diff=0.1):
    diff = perturbed - original
    clipped_diff = np.clip(diff, -max_diff, max_diff)
    return original + clipped_diff

# Function to load and classify an image
def classify_image(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return

        img = Image.open(image_path).convert('RGB')
        
        # Preprocess the image
        inputs = feature_extractor(images=img, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        
        # Normalize pixel values to [0, 1] range
        pixel_values = (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min())
        
        with torch.no_grad():
            outputs = model(pixel_values)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            true_label = model.config.id2label[predicted_class_idx]
        
        # Display the original image with true label
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Original Image\nPredicted class: {true_label}")
        plt.axis('off')

        # Generate adversarial example using DeepFool
        pixel_values_np = pixel_values.numpy()
        deepfool = DeepFool(classifier=art_classifier, max_iter=50, epsilon=0.02, nb_grads=10)
        adv_inputs = deepfool.generate(x=pixel_values_np)
        
        # Apply additional perturbation limiting
        adv_inputs = limit_perturbation(pixel_values_np, adv_inputs, max_diff=0.1)
        
        adv_inputs_tensor = torch.tensor(adv_inputs).float()

        # Classify the adversarial image
        with torch.no_grad():
            adv_outputs = model(pixel_values=adv_inputs_tensor)
            adv_logits = adv_outputs.logits
            adv_predicted_class_idx = adv_logits.argmax(-1).item()
            adv_label = model.config.id2label[adv_predicted_class_idx]

        # Convert adversarial image back to PIL format
        adv_image_np = adv_inputs_tensor.squeeze().permute(1, 2, 0).numpy()
        adv_image = Image.fromarray((adv_image_np * 255).astype(np.uint8))

        # Display the adversarial image with misclassification label
        plt.subplot(1, 2, 2)
        plt.imshow(adv_image)
        plt.title(f"Adversarial Image (DeepFool)\nPredicted class: {adv_label}")
        plt.axis('off')

        plt.show()

    except Exception as e:
        print(f"Error loading or processing image: {e}")
        import traceback
        print(traceback.format_exc())

# Main loop to ask for image file names
images_folder = "images"
print("Place your images in the 'images' folder.")
while True:
    try:
        image_name = input("Enter the image file name (or press Ctrl+C to exit): ")
        image_path = os.path.join(images_folder, image_name)
        classify_image(image_path)
    except KeyboardInterrupt:
        print("\nExiting...")
        break