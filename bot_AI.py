import telebot
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from dotenv import load_dotenv
import timm
import json

# Загружаем токен бота
load_dotenv()


bot = telebot.TeleBot(str(os.getenv("API")) + ":" + str(os.getenv("API2")))

# Загружаем конфигурацию из файла
CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

IMG_SIZE = config["img_size"]
BACKBONE = config["backbone"]
NUM_CLASSES = config["num_classes"]  # ВАЖНО: это ключевое значение!

# Точная архитектура модели (как в train.py)
class PlantModel(nn.Module):
    def __init__(self, backbone="efficientnet_b0", num_classes=3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, 
            pretrained=False, 
            num_classes=0, 
            global_pool="avg"
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Linear(in_features, num_classes)
        self.regressor = nn.Linear(in_features, 1)  # для высоты, но мы не будем использовать

    def forward(self, x):
        feats = self.backbone(x)
        cls_out = self.classifier(feats)
        reg_out = torch.sigmoid(self.regressor(feats))
        return cls_out, reg_out

# Загружаем модель
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantModel(backbone=BACKBONE, num_classes=NUM_CLASSES).to(device)

MODEL_PATH = "plant_model.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Трансформации изображения
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Маппинг классов
label_map = {0: "not_plant", 1: "plant_healthy", 2: "plant_unhealthy"}

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        cls_out, reg_out = model(tensor)
        cls_idx = cls_out.argmax(dim=1).item()
        confidence = torch.softmax(cls_out, dim=1)[0][cls_idx].item()
    
    label = label_map[cls_idx]
    return label, confidence

@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(message, "🌱 Привет! Отправь фото растения — я скажу:\n- Это растение или нет?\n- Здоровое оно или больное?")

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded = bot.download_file(file_info.file_path)
        file_path = f"temp_{message.chat.id}.jpg"
        
        with open(file_path, "wb") as f:
            f.write(downloaded)
        
        label, confidence = predict(file_path)
        os.remove(file_path)
        
        if label == "not_plant":
            response = "❌ Это не растение. Попробуйте сфотографировать растение поближе."
        else:
            status = "здоровое" if label == "plant_healthy" else "больное"
            response = (
                f"✅ Это растение!\n"
                f"Состояние: {status}\n"
                f"Уверенность: {confidence:.1%}"
            )
        
        bot.reply_to(message, response)
    
    except Exception as e:
        bot.reply_to(message, f"⚠️ Ошибка: {str(e)}\nПопробуйте другое фото.")

print("✅ Бот запущен! Жду фото...")
bot.polling(none_stop=True)