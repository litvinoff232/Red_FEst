import os
import pandas as pd
import random

# Пути относительно проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "plant_dataset")
CSV_PATH = os.path.join(BASE_DIR, "dataset.csv")

# Классы (папки внутри plant_dataset/train и plant_dataset/val)
CLASSES = ["plant_healthy", "plant_unhealthy", "not_plant"]

def create_dataset_csv():
    rows = []
    for split in ["train", "val"]:
        split_dir = os.path.join(DATASET_DIR, split)
        if not os.path.exists(split_dir):
            print(f"⚠️ Папка {split_dir} не найдена, пропускаю")
            continue

        for label in CLASSES:
            label_dir = os.path.join(split_dir, label)
            if not os.path.exists(label_dir):
                print(f"⚠️ Папка {label_dir} не найдена, пропускаю")
                continue

            for fname in os.listdir(label_dir):
                fpath = os.path.join(label_dir, fname)
                if not os.path.isfile(fpath):
                    continue

                # Случайная высота для примера (замени на свои данные)
                height_cm = random.randint(30, 120)

                rows.append({
                    "image_path": fpath,
                    "label": label,
                    "split": split,
                    "height_cm": height_cm
                })

    df = pd.DataFrame(rows)

    # Удаляем битые пути
    df = df[df["image_path"].apply(os.path.exists)]

    print(f"✅ dataset.csv создан: {len(df)} изображений")
    print(df.head())

    # Статистика
    print(df["label"].value_counts())
    print(df["split"].value_counts())

    df.to_csv(CSV_PATH, index=False)

if __name__ == "__main__":
    create_dataset_csv()
