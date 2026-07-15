
import os
import glob
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Paths
SRC_ROOT = "../data/MultiDisaster_20250320/UAV"
DST_ROOT = "../data/MultiDisaster_20250320/UAV_preprocessed"

# Ensure output root exists
os.makedirs(DST_ROOT, exist_ok=True)

# Crop and resize config
CROP_SIZE = 768
RESIZE_TO = 384


def process_folder(folder_name):
    src_folder = os.path.join(SRC_ROOT, folder_name)
    dst_folder = os.path.join(DST_ROOT, folder_name)
    os.makedirs(dst_folder, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(src_folder, "*.jpg")))

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size  # expected to be (1024, 768)

            # Center crop to 768x768
            left = (w - CROP_SIZE) // 2
            top = 0
            right = left + CROP_SIZE
            bottom = top + CROP_SIZE
            img_cropped = img.crop((left, top, right, bottom))

            # Resize to 384x384
            img_resized = img_cropped.resize((RESIZE_TO, RESIZE_TO), Image.BILINEAR)

            # Save to destination
            filename = os.path.basename(img_path)
            save_path = os.path.join(dst_folder, filename)
            img_resized.save(save_path, quality=95)

        except Exception as e:
            print(f"[{folder_name}] Error processing {img_path}: {e}")


def main():
    uav_folders = [f for f in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, f))]

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_folder, uav_folders), total=len(uav_folders), desc="Processing UAV"))


if __name__ == "__main__":
    main()


