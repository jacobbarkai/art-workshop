import os
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from art.attacks.evasion import DeepFool
from art.estimators.classification import PyTorchClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD

# Suppress warnings
warnings.filterwarnings("ignore")

# Wrapper class for the model
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model
        self.classifier = torch.nn.Linear(1000, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        x = self.features(x)
        if isinstance(x, tuple):
            x = x[0]
        if hasattr(x, 'logits'):
            x = x.logits
        return self.classifier(x)

# Function to create ART classifier
def create_art_classifier(model):
    wrapped_model = ModelWrapper(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.001)

    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=10,
        clip_values=(0, 1),
    )
    return wrapped_model, classifier

# Function to load CIFAR-10 subset for training
def load_cifar10_subset(num_samples=1000):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    subset = torch.utils.data.Subset(dataset, range(num_samples))
    return DataLoader(subset, batch_size=32, shuffle=True)

# Function to load CIFAR-10 test set
def load_cifar10_test(num_samples=1000):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    subset = torch.utils.data.Subset(dataset, range(num_samples))
    return DataLoader(subset, batch_size=32, shuffle=False)

# Function to perform adversarial training
from PIL import Image

def adversarial_training(model, art_classifier, train_loader, num_epochs=3):
    trainer = AdversarialTrainerMadryPGD(
        classifier=art_classifier,
        nb_epochs=num_epochs,
        batch_size=32,
        eps=0.03,
        eps_step=0.01,
        max_iter=10,
        num_random_init=1
    )

    # Convert the DataLoader to numpy arrays
    x_train = []
    y_train = []
    for data, target in train_loader:
        x_train.append(data.numpy())
        y_train.append(target.numpy())
    
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Ensure x_train has the correct shape (N, 3, 224, 224)
    if x_train.shape[1:] != (3, 224, 224):
        x_train = np.transpose(x_train, (0, 2, 3, 1))
        x_train = np.array([
            np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((224, 224))) 
            for img in x_train
        ])
        x_train = np.transpose(x_train, (0, 3, 1, 2)) / 255.0

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Perform adversarial training
    trainer.fit(x_train, y_train)

    print("Adversarial training completed.")

# Function to test model robustness
def test_model_robustness(model, art_classifier, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    clean_correct = 0
    adv_correct = 0
    total = 0

    deepfool = DeepFool(classifier=art_classifier, max_iter=50, epsilon=0.02, nb_grads=10)

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)

        # Clean accuracy
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            clean_correct += predicted.eq(labels).sum().item()

        # Generate adversarial examples
        adv_images = deepfool.generate(x=images.cpu().numpy())
        adv_images = torch.tensor(adv_images).float().to(device)

        # Adversarial accuracy
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, adv_predicted = adv_outputs.max(1)
            adv_correct += adv_predicted.eq(labels).sum().item()

    clean_accuracy = 100. * clean_correct / total
    adv_accuracy = 100. * adv_correct / total

    print(f"Clean Accuracy: {clean_accuracy:.2f}%")
    print(f"Adversarial Accuracy: {adv_accuracy:.2f}%")
    print(f"Robustness (Adv Acc / Clean Acc): {adv_accuracy / clean_accuracy:.2f}")

    return clean_accuracy, adv_accuracy

# Function to limit perturbation
def limit_perturbation(original, perturbed, max_diff=0.1):
    diff = perturbed - original
    clipped_diff = np.clip(diff, -max_diff, max_diff)
    return original + clipped_diff

# Function to load and classify an image
def classify_image(image_path, model, art_classifier):
    try:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return

        img = Image.open(image_path).convert('RGB')
        
        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        pixel_values = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(pixel_values)
            predicted_class_idx = outputs.argmax(-1).item()
            true_label = datasets.CIFAR10.classes[predicted_class_idx]
        
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
            adv_outputs = model(adv_inputs_tensor)
            adv_predicted_class_idx = adv_outputs.argmax(-1).item()
            adv_label = datasets.CIFAR10.classes[adv_predicted_class_idx]

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

# Main execution
if __name__ == "__main__":
    try:
        # Load the pre-trained model from Hugging Face
        model_name = "microsoft/resnet-50"
        model = AutoModelForImageClassification.from_pretrained(model_name)

        print("Original model architecture:")
        print(model)

        # Create ART classifier
        wrapped_model, art_classifier = create_art_classifier(model)

        print("\nModified model architecture:")
        print(wrapped_model)

        # Load CIFAR-10 subset for training
        train_loader = load_cifar10_subset()

        # Print sample input and output
        sample_input, _ = next(iter(train_loader))
        print(f"\nSample input shape: {sample_input.shape}")
        with torch.no_grad():
            sample_output = wrapped_model(sample_input)
        print(f"Sample output shape: {sample_output.shape}")

        # Perform adversarial training
        adversarial_training(wrapped_model, art_classifier, train_loader)

        # Load CIFAR-10 test set
        test_loader = load_cifar10_test()

        # Test model robustness
        print("Testing model robustness...")
        clean_acc, adv_acc = test_model_robustness(wrapped_model, art_classifier, test_loader)

        # Main loop to ask for image file names
        images_folder = "images"
        print("\nPlace your CIFAR-10 test images in the 'images' folder.")
        while True:
            try:
                image_name = input("Enter the image file name (or press Ctrl+C to exit): ")
                image_path = os.path.join(images_folder, image_name)
                classify_image(image_path, wrapped_model, art_classifier)
            except KeyboardInterrupt:
                print("\nExiting...")
                break

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print(traceback.format_exc())