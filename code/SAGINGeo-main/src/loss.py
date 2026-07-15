import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, features_dict, logit_scale):
        # Normalize all feature sets
        for key in features_dict:
            features_dict[key] = F.normalize(features_dict[key], dim=-1)

        views = list(features_dict.keys())
        total_loss = 0.0
        count = 0

        labels = torch.arange(next(iter(features_dict.values())).size(0), dtype=torch.long, device=self.device)

        # Iterate over all unique pairs (i, j), i != j
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                feat_i = features_dict[views[i]]
                feat_j = features_dict[views[j]]

                # Compute logits
                logits_ij = logit_scale * feat_i @ feat_j.T
                logits_ji = logits_ij.T

                # Symmetric InfoNCE loss
                loss = (self.loss_function(logits_ij, labels) + self.loss_function(logits_ji, labels)) / 2.0
                total_loss += loss
                count += 1

        return total_loss / count if count > 0 else 0.0
