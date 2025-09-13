import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import timm
from tqdm import tqdm


DATASET_PATH = "dataset.csv"
MODEL_PATH = "plant_model.pth"
CONFIG_PATH = "config.json"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlantDataset(Dataset):
    def __init__(self, df, transform=None, max_height=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        if max_height is None:
            self.max_height = self.df["height_cm"].max()
        else:
            self.max_height = max_height

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label_map = {"not_plant": 0, "plant_healthy": 1, "plant_unhealthy": 2}
        label = label_map[row["label"]]

        
        height = row["height_cm"] / self.max_height

        return img, torch.tensor(label, dtype=torch.long), torch.tensor(height, dtype=torch.float32)


class PlantModel(nn.Module):
    def __init__(self, backbone="efficientnet_b0", num_classes=3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        in_features = self.backbone.num_features
        self.classifier = nn.Linear(in_features, num_classes)
        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        feats = self.backbone(x)
        cls_out = self.classifier(feats)
        reg_out = torch.sigmoid(self.regressor(feats))  # нормализованный рост [0,1]
        return cls_out, reg_out


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


df = pd.read_csv(DATASET_PATH)
print(f"📂 Загружено строк: {len(df)}")

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]

train_dataset = PlantDataset(train_df, transform=transform)
val_dataset = PlantDataset(val_df, transform=transform, max_height=train_dataset.max_height)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = PlantModel().to(DEVICE)
criterion_cls = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels, heights in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
        imgs, labels, heights = imgs.to(DEVICE), labels.to(DEVICE), heights.to(DEVICE)

        optimizer.zero_grad()
        cls_out, reg_out = model(imgs)
        loss_cls = criterion_cls(cls_out, labels)
        loss_reg = criterion_reg(reg_out.squeeze(), heights)
        loss = loss_cls + loss_reg
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for imgs, labels, heights in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]"):
            imgs, labels, heights = imgs.to(DEVICE), labels.to(DEVICE), heights.to(DEVICE)
            cls_out, reg_out = model(imgs)
            loss_cls = criterion_cls(cls_out, labels)
            loss_reg = criterion_reg(reg_out.squeeze(), heights)
            loss = loss_cls + loss_reg
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"📉 Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


torch.save(model.state_dict(), MODEL_PATH)

config = {
    "max_height": int(train_dataset.max_height),
    "img_size": int(IMG_SIZE),
    "backbone": "efficientnet_b0"
}
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f)

print(f"✅ Модель сохранена: {MODEL_PATH}")
print(f"✅ Конфиг сохранён: {CONFIG_PATH}")
