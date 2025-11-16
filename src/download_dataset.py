# src/download_dataset.py
import kagglehub
import shutil
import os
import pandas as pd

# 1️⃣ Download latest version of Sentiment140 dataset
print("⏳ Downloading Sentiment140 dataset...")
path = kagglehub.dataset_download("kazanova/sentiment140")
print("✅ Download complete!")

print("Path to dataset files:", path)

# 2️⃣ Find the main CSV file
csv_file = os.path.join(path, "training.1600000.processed.noemoticon.csv")

# 3️⃣ Copy it into your project's /data folder
dest_path = os.path.join("data", "tweets.csv")
shutil.copy(csv_file, dest_path)

print(f"✅ Copied dataset to {dest_path}")

# 4️⃣ (Optional) Preview first few rows
df = pd.read_csv(dest_path, encoding='latin-1', header=None)
df = df.rename(columns={0: "target", 5: "text"})
print(df[["target", "text"]].head())
