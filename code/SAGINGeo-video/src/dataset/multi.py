import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import av
from transformers import VivitImageProcessor
import numpy as np
import av
from PIL import Image

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    Uniformly sample `clip_len` frames from the second half of the video.

    Args:
        clip_len (int): Number of frames to sample.
        frame_sample_rate (int): Not used here directly since we go uniform.
        seg_len (int): Total number of frames in the video.

    Returns:
        indices (np.ndarray): Array of frame indices (int64).
    """
    # Define start and end for the second half
    start_idx = seg_len // 2
    end_idx = seg_len - 1

    if end_idx - start_idx + 1 < clip_len:
        # Not enough frames in second half: fallback to full range
        indices = np.linspace(0, seg_len - 1, num=clip_len)
    else:
        indices = np.linspace(start_idx, end_idx, num=clip_len)

    return np.round(indices).astype(np.int64)

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    return [Image.fromarray(f) for f in frames]

class MultiDatasetTrain(Dataset):
    def __init__(self, data_folder, task=None, split=80,
                 transforms_square=None, transforms_wide=None,
                 img_size=256, vivit_processor=None, num_views=8):
        super().__init__()
        self.data_folder = data_folder
        self.task = task
        self.transforms_square = transforms_square
        self.transforms_wide = transforms_wide
        self.img_size = img_size
        self.vivit_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.num_views = num_views
        csv_file = f"{split}%train_RSI_SVI_UAV_VGI.csv"
        self.df = pd.read_csv(os.path.join(data_folder, "RSI_SVI_UAV_VGI", csv_file), header=None)
        self.samples = self.df.values.tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        keys = ["RSI", "SVI", "UAV", "VGI"]
        sample = {}

        for i, key in enumerate(keys):
            if key not in self.task:
                continue

            img_path = os.path.normpath(os.path.join(self.data_folder, row[i].replace('\\', '/')))

            if key == "UAV":
                uav_id = os.path.splitext(os.path.basename(img_path))[0]
                video_path = os.path.join(self.data_folder, "Other", "UAV_video", f"{uav_id}.m4v")

                try:
                    container = av.open(video_path)
                    num_frames = container.streams.video[0].frames
                    target_views = self.num_views  # Can be 2, 4, 8, 16, 32

                    # Sample the desired number of distinct frames
                    indices = sample_frame_indices(clip_len=target_views, frame_sample_rate=1, seg_len=num_frames)
                    frames = read_video_pyav(container, indices)
                    container.close()

                    if not frames:
                        raise ValueError("No frames after sampling.")

                    # Repeat each frame (in blocks) to reach exactly 32 frames
                    repeat_factor = 32 // target_views
                    repeated_frames = []
                    for frame in frames:
                        repeated_frames.extend([frame] * repeat_factor)

                    processed = self.vivit_processor(repeated_frames, return_tensors="pt")
                    sample[key] = processed["pixel_values"].squeeze(0)

                except Exception:
                    print("Error in video input")
                    sample[key] = torch.zeros((3, 32, self.img_size, self.img_size), dtype=torch.float32)


            else:
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)

                if key in ["RSI", "VGI"] and self.transforms_square:
                    img_tensor = self.transforms_square(image=img_np)['image']
                elif key == "SVI" and self.transforms_wide:
                    img_tensor = self.transforms_wide(image=img_np)['image']
                else:
                    img_tensor = torch.tensor(img_np).permute(2, 0, 1).float() / 255.0

                sample[key] = img_tensor

        return sample

class MultiDatasetEval(Dataset):
    def __init__(self, data_folder, task=None, split=20,
                 transforms_square=None, transforms_wide=None,
                 img_size=256, vivit_processor=None, train=False, num_views=8):
        super().__init__()
        self.data_folder = data_folder
        self.task = task
        self.transforms_square = transforms_square
        self.transforms_wide = transforms_wide
        self.num_views = 8
        self.img_size = img_size
        self.vivit_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

        csv_file = f"{split}%test_RSI_SVI_UAV_VGI.csv"
        self.df = pd.read_csv(os.path.join(data_folder, "RSI_SVI_UAV_VGI", csv_file), header=None)
        self.samples = self.df.values.tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        keys = ["RSI", "SVI", "UAV", "VGI"]
        sample = {}

        for i, key in enumerate(keys):
            if key not in self.task:
                continue

            img_path = os.path.normpath(os.path.join(self.data_folder, row[i].replace('\\', '/')))

            if key == "UAV":
                uav_id = os.path.splitext(os.path.basename(img_path))[0]
                video_path = os.path.join(self.data_folder, "Other", "UAV_video", f"{uav_id}.m4v")

                try:
                    container = av.open(video_path)
                    num_frames = container.streams.video[0].frames
                    target_views = self.num_views  # Can be 2, 4, 8, 16, 32

                    # Sample the desired number of distinct frames
                    indices = sample_frame_indices(clip_len=target_views, frame_sample_rate=1, seg_len=num_frames)
                    frames = read_video_pyav(container, indices)
                    container.close()

                    if not frames:
                        raise ValueError("No frames after sampling.")

                    # Repeat each frame (in blocks) to reach exactly 32 frames
                    repeat_factor = 32 // target_views
                    repeated_frames = []
                    for frame in frames:
                        repeated_frames.extend([frame] * repeat_factor)

                    processed = self.vivit_processor(repeated_frames, return_tensors="pt")
                    sample[key] = processed["pixel_values"].squeeze(0)

                except Exception:
                    print("Error in video input")
                    sample[key] = torch.zeros((3, 32, self.img_size, self.img_size), dtype=torch.float32)

            else:
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)

                if key in ["RSI", "VGI"] and self.transforms_square:
                    img_tensor = self.transforms_square(image=img_np)['image']
                elif key == "SVI" and self.transforms_wide:
                    img_tensor = self.transforms_wide(image=img_np)['image']
                else:
                    img_tensor = torch.tensor(img_np).permute(2, 0, 1).float() / 255.0

                sample[key] = img_tensor

        label = torch.tensor(index, dtype=torch.long)
        return sample, label


