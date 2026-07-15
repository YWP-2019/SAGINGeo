import os
import time
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
import pickle
from src.dataset.multi import MultiDatasetEval, MultiDatasetTrain
from src.transforms import get_transforms_train, get_transforms_val
from src.utils import setup_system, Logger
from src.trainer import train
from src.evaluate.multi import evaluate
from src.loss import InfoNCE
from src.model import TimmModel


@dataclass
class Configuration:

    # Model
    model: str = 'vit_large_patch14_dinov2.lvd142m'# 'timm/vit_base_patch14_dinov2.lvd142m'#'timm/convnext_tiny.in12k_ft_in1k'

    # Override model image size
    img_size: int = 384
    task: tuple = ('RSI', 'UAV', 'VGI', 'SVI')  # "RSI", "SVI", "UAV", "VGI"
    split: int = 80
    extract_train: bool = False
    # Training
    mixed_precision: bool = True
    seed = 42
    verbose: bool = True
    gpu_ids: tuple = (0,)   # GPU ids for training

    # Eval
    batch_size_eval: int = 32
    normalize_features: bool = True

    # Dataset
    data_folder = "./data/MultiDisaster_20250320/"

    # Savepath for model checkpoints
    model_path: str = "./multidisaster_model"

    # Eval before training
    zero_shot: bool = False

    # Checkpoint to start from
    checkpoint_start = "multidisaster_model/timm/vit_large_patch14_dinov2.lvd142m/084527/weights_end.pth"

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else len(gpu_ids) * 8

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True

    # make cudnn deterministic
    cudnn_deterministic: bool = False


# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#

config = Configuration()


if __name__ == '__main__':

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#

    print("\nModel: {}".format(config.model))

    model = TimmModel(config.model,
                      pretrained=True,
                      img_size=config.img_size)

    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size

    image_size_sat = (img_size, img_size)

    new_width = config.img_size * 2
    new_hight = round((512 / 1024) * new_width)
    img_size_ground = (new_hight, new_width)

    # Load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    # Model to device
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )

    dataset_test = MultiDatasetEval(data_folder=config.data_folder,
                                    transforms_square=sat_transforms_val,
                                    transforms_wide=ground_transforms_val,
                                    task=config.task,
                                    split=100 - config.split,
                                    img_size=config.img_size,
                                    train=config.extract_train
                                    )

    dataloader_test = DataLoader(dataset_test,
                                 batch_size=config.batch_size_eval,
                                 num_workers=config.num_workers,
                                 shuffle=False,
                                 pin_memory=True)

    print("Images Test:", len(dataset_test))


    r1_test, features_dict = evaluate(config=config,
                            model=model,
                           dataloader=dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

    features_dict_cpu = {k: v.cpu() for k, v in features_dict.items()}

    # Save to a pickle file
    if config.extract_train:
        with open("features_train.pkl", "wb") as f:
            pickle.dump(features_dict_cpu, f)
    else:
        with open("features_test.pkl", "wb") as f:
            pickle.dump(features_dict_cpu, f)
