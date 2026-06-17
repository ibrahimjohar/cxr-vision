import os
import pydicom
import numpy as np
from PIL import Image

src = "data/raw/rsna/stage_2_train_images"
dst = "data/raw/rsna/stage_2_train_images_png"
os.makedirs(dst, exist_ok=True)

files = [f for f in os.listdir(src) if f.endswith(".dcm")]
print(f"converting {len(files)} dicom files...")

for i, f in enumerate(files):
    arr = pydicom.dcmread(os.path.join(src, f)).pixel_array.astype(np.float32)
    arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(dst, f.replace(".dcm", ".png")))
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(files)}")

print("done.")