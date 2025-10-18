import os
import torch
from PIL import Image

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34


# Path to the folder with images, structure - 
# folder_name/class_x/image_xx/image_xx.png

# image_folder = 'downloaded_images_asses'
image_folder = 'path_severity'
# image_folder = 'screening0123'
# image_folder = '2b6k'



# Load pre-trained ResNet-18 model
# model = resnet18(pretrained=True)
model = resnet34(pretrained=True)
model.eval()

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256), # 256
    transforms.CenterCrop(224), #???????????
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1979, 0.1979, 0.1979], std=[0.1969, 0.1969, 0.1969]),
])

full_dataset = datasets.ImageFolder(root=image_folder, transform=preprocess)
num_classes = len(full_dataset.classes)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Replace the final layer to match the number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

#  train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs =20
print("Training ResNet...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluate the model
from sklearn.metrics import roc_auc_score
model.eval()
correct, total = 0, 0
all_labels = []
all_probs = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)  # Get probabilities
        if num_classes == 2:
            # For binary classification, get probability for class 1
            class1_probs = probs[:, 1].cpu().numpy()
            all_probs.extend(class1_probs)
        else:
            # For multiclass, use probabilities for all classes
            all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")
# ROC AUC calculation
if num_classes == 2:
    auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC AUC: {auc:.4f}")
else:
    # For multiclass, use 'ovr' (one-vs-rest) strategy
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    print(f"Multiclass ROC AUC: {auc:.4f}")

# # Compute dataset mean and std for normalization (used for pathology severity "path_severity" images, turned out mean [0.1979, 0.1979, 0.1979], std [0.1969, 0.1969, 0.1969])
# def compute_normalization(dataset_path):
#     # Compute mean and std for normalization
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ])
#     dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
#     loader = DataLoader(dataset, batch_size=100, shuffle=False)

#     mean = 0.0
#     std = 0.0
#     total_images = 0

#     for images, _ in loader:
#         batch_samples = images.size(0)
#         images = images.view(batch_samples, images.size(1), -1)
#         mean += images.mean(2).sum(0)
#         std += images.std(2).sum(0)
#         total_images += batch_samples

#     mean /= total_images
#     std /= total_images

#     return mean, std
# mean, std = compute_normalization(image_folder)
# print("Path Severity Dataset Mean: ", mean, " | Std: ", std)