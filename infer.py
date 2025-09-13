import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm


MODEL_PATH = "plant_model.pth"
CONFIG_PATH = "config.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlantModel(nn.Module):
    def __init__(self, backbone="efficientnet_b0", num_classes=3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
        in_features = self.backbone.num_features
        self.classifier = nn.Linear(in_features, num_classes)
        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        feats = self.backbone(x)
        cls_out = self.classifier(feats)
        reg_out = torch.sigmoid(self.regressor(feats)) 
        return cls_out, reg_out

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

IMG_SIZE = config["img_size"]
MAX_HEIGHT = config["max_height"]
BACKBONE = config["backbone"]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


model = PlantModel(backbone=BACKBONE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

label_map = {0: "not_plant", 1: "plant_healthy", 2: "plant_unhealthy"}


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cls_out, reg_out = model(img)
        cls_idx = cls_out.argmax(dim=1).item()
        label = label_map[cls_idx]
        height_norm = reg_out.item()
        height_cm = height_norm * MAX_HEIGHT

    return {
        "path": img_path,
        "label": label,
        "height_cm": round(height_cm, 2)
    }

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("⚠️ Использование: python infer.py <путь к изображению или папке>")
    else:
        target = sys.argv[1]
        if os.path.isdir(target):
            for fname in os.listdir(target):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    result = predict_image(os.path.join(target, fname))
                    print(result)
        else:
            result = predict_image(target)
            print(result)
