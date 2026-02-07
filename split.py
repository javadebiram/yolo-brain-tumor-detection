#split (80٪ / 20٪)
import os
import random
import shutil

base = r"C:\Users\javad\Desktop\project\dataset" #مسیر اصلی فایل های ذخیره شده
img_train = os.path.join(base, "images/train")
lbl_train = os.path.join(base, "labels/train")

img_val = os.path.join(base, "images/val")
lbl_val = os.path.join(base, "labels/val")

os.makedirs(img_val, exist_ok=True)
os.makedirs(lbl_val, exist_ok=True)

files = os.listdir(img_train)
random.shuffle(files)

val_count = int(0.2 * len(files))

for f in files[:val_count]:
    shutil.move(os.path.join(img_train, f), os.path.join(img_val, f))
    shutil.move(os.path.join(lbl_train, f.replace(".png", ".txt")),
                os.path.join(lbl_val, f.replace(".png", ".txt")))

print("Split done.")