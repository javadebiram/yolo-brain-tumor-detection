import nibabel as nib
import numpy as np
import cv2
import os

flair_path = r"C:/Users/javad/Desktop/MRI dataset/Brats18_2013_10_1/Brats18_2013_10_1_flair.nii" #مسیر فایل flair
seg_path   = r"C:/Users/javad/Desktop/MRI dataset/Brats18_2013_10_1/Brats18_2013_10_1_seg.nii" #مسیر فایل segment

img_out = r"C:\Users\javad\Desktop\main project\main\main dataset\images" #عکس ها در این ادرس ذخیره میشوند
lbl_out = r"C:\Users\javad\Desktop\main project\main\main dataset\val" #مختصات عکس ها در این ادرس ذخیره میشوند

os.makedirs(img_out, exist_ok=True)
os.makedirs(lbl_out, exist_ok=True)

flair = nib.load(flair_path).get_fdata()
seg   = nib.load(seg_path).get_fdata()

# normalize flair
flair = (flair - flair.min()) / (flair.max() - flair.min())
flair = (flair * 255).astype(np.uint8)


idx = 0
for i in range(flair.shape[2]):
    mask = seg[:, :, i]
    if np.sum(mask) == 0:
        continue

    img = flair[:, :, i]

    # whole tumor
    mask_bin = (mask > 0).astype(np.uint8) * 255


    # استخراج کانتورها
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS) # یا CHAIN_APPROX_SIMPLE

    H, W = img.shape

    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        
        
        epsilon = 0.005 * cv2.arcLength(cnt, True) # درصد کوچکی از محیط
        approx = cv2.approxPolyDP(cnt, epsilon, True) # True یعنی کانتور بسته است

        cnt_simplified = approx.squeeze() # نقاط ساده شده

        if cnt_simplified.ndim != 2:
            continue

        poly = []
        for x, y in cnt_simplified:
            poly.append(x / W)
            poly.append(y / H)

        polygons.append(poly)
    



    if not polygons:
        continue

    name = f"slice_{idx:04d}"
    cv2.imwrite(os.path.join(img_out, name + ".png"), img)

    with open(os.path.join(lbl_out, name + ".txt"), "w") as f:
        for poly in polygons:
            f.write("0 " + " ".join(map(str, poly)) + "\n")

    idx += 1

print("YOLO Segmentation dataset created.")