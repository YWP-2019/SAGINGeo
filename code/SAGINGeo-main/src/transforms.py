import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms_train(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0):

    satellite_transforms = A.Compose([
        A.ImageCompression(quality_range=(90, 100), p=0.5),
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(num_holes_range=(10, 25),
                            hole_height_range=(0.1, 0.2),
                            hole_width_range=(0.1, 0.2),
                            p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    ground_transforms = A.Compose([
        A.ImageCompression(quality_range=(90, 100), p=0.5),
        A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.5, p=1.0),
            A.CoarseDropout(num_holes_range=(10, 25),
                            hole_height_range=(0.1, 0.2),
                            hole_width_range=(0.1, 0.2),
                            p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    return satellite_transforms, ground_transforms


def get_transforms_val(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0):

    satellite_transforms = A.Compose([A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])

    ground_transforms = A.Compose([
        A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    return satellite_transforms, ground_transforms
