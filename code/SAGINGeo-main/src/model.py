import torch
import torch.nn as nn
import timm
import numpy as np

class TimmModel(nn.Module):

    def __init__(self, model_name, pretrained=True, img_size=383, embed_dim=1024):
        super().__init__()

        self.img_size = img_size
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if "vit" in model_name.lower():
            new_width = img_size * 2
            new_height = round((512 / 1024) * new_width)
            self.img_size_wide = (new_height, new_width)

            self.model_square = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=(img_size, img_size))
            self.model_wide = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=self.img_size_wide)
            self.model_uav = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=(img_size, img_size))

        elif "convnext" in model_name.lower():
            self.model_main = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.model_uav = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)


    def get_config(self):
        if hasattr(self, "model"):
            return timm.data.resolve_model_data_config(self.model)
        elif hasattr(self, "model_main"):
            return timm.data.resolve_model_data_config(self.model_main)
        else:
            return timm.data.resolve_model_data_config(self.model_square)


    def forward(self, data_dict):
        features_dict = {}

        for key, img in data_dict.items():
            features = self._forward_single(img, key)
            features_dict[key] = features

        return features_dict


    def _forward_single(self, x, key):
        if hasattr(self, "model"):
            return self.model(x)

        if "vit" in self.model_name.lower():
            if key == "SVI":
                return self.model_wide(x)
            elif key == "UAV":
                return self.model_uav(x)
            elif key in ["RSI", "VGI"]:
                return self.model_square(x)
            else:
                raise ValueError(f"Unknown view type '{key}' for ViT model.")

        elif "convnext" in self.model_name.lower():
            if key == "UAV":
                return self.model_uav(x)
            else:
                return self.model_main(x)

        else:
            raise ValueError(f"Unknown model type in '{self.model_name}'")
