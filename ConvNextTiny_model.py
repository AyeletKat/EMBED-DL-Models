
# use convnext_tiny model from torchvision.models with pretrained weights

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

dataset_path = "C:/emory/embed/path_severity"

transform = transforms.Compose([
    transforms.Resize((224, 224)),   # ConvNeXt default input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1979, 0.1979, 0.1979], std=[0.1969, 0.1969, 0.1969]),
])

full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
num_classes = len(full_dataset.classes)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# using ConvNeXt Model
weights = ConvNeXt_Tiny_Weights.DEFAULT
model = convnext_tiny(weights=weights)

# replace of classifier head
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
EPOCHS = 15
print("Training ConvNextTiny...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
from sklearn.metrics import roc_auc_score

model.eval()

correct, total = 0, 0
all_labels = []
all_probs = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
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
