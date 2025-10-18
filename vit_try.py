"""
October 2025

This is the initial code for the Vision Transformer (ViT) model with BIRADS digit taken embedded as tabular data.
The used version with full code appears in the colab notebook "EMBED-ViT&BIRADS_model.ipynb" also attached in the github repository.

"""



import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import roc_auc_score

# --- Custom Dataset ---
class SeverityBiradsDataset(torch.utils.data.Dataset):
    # Map BIRADS letter to number
    birads_map = {'A': 0, 'N': 1, 'B': 2, 'P': 3, 'S': 4, 'M': 5, 'K': 6}

    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(root_dir)))}
        for class_name in sorted(os.listdir(root_dir)):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for img_folder in os.listdir(class_folder):
                img_folder_path = os.path.join(class_folder, img_folder)
                if not os.path.isdir(img_folder_path):
                    continue
                for fname in os.listdir(img_folder_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(img_folder_path, fname)
                        birads_letter = fname[0]
                        birads_digit = self.birads_map.get(birads_letter, -1)  # -1 if not found
                        label = self.class_to_idx[class_name]
                        self.samples.append((img_path, float(birads_digit), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, birads_digit, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        birads_tensor = torch.tensor([birads_digit], dtype=torch.float32)
        return img, birads_tensor, label
    
# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1979, 0.1979, 0.1979], std=[0.1969, 0.1969, 0.1969]),
])

image_folder = 'path_severity'  # Change to your dataset folder
full_dataset = SeverityBiradsDataset(image_folder, transform=preprocess)
num_classes = len(full_dataset.class_to_idx)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# --- ViT + Tabular Fusion Model ---
class ViTWithBirads(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()  # Remove default head
        self.tabular = nn.Linear(1, 32)  # For BIRADS digit
        self.classifier = nn.Linear(self.vit.num_features + 32, num_classes)

    def forward(self, x_img, x_birads):
        img_feat = self.vit(x_img)
        tab_feat = self.tabular(x_birads)
        combined = torch.cat([img_feat, tab_feat], dim=1)
        out = self.classifier(combined)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTWithBirads(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# --- Training Loop ---
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, birads, labels in train_loader:
        images = images.to(device)
        birads = birads.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, birads)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# --- Evaluation ---
model.eval()
correct, total = 0, 0
all_labels = []
all_probs = []
with torch.no_grad():
    for images, birads, labels in test_loader:
        images = images.to(device)
        birads = birads.to(device)
        labels = labels.to(device)
        outputs = model(images, birads)
        probs = torch.softmax(outputs, dim=1)
        if num_classes == 2:
            class1_probs = probs[:, 1].cpu().numpy()
            all_probs.extend(class1_probs)
        else:
            all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")
if num_classes == 2:
    auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC AUC: {auc:.4f}")
else:
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    print(f"Multiclass ROC AUC: {auc:.4f}")