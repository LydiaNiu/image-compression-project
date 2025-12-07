import os
import random

folder = "/Users/rjiwookim/image-compression-project/originals/Urban"

# image file extensions to include
image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

# get all images
images = [
    f for f in os.listdir(folder)
    if os.path.splitext(f)[1].lower() in image_exts
]

# shuffle them randomly
random.shuffle(images)

# rename them sequentially: port1.jpg, port2.jpg, ...
for i, filename in enumerate(images, start=1):
    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, f"urb{i}.jpg")   # always .jpg
    os.rename(old_path, new_path)

print("Done!")
