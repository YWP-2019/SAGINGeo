import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.amp import autocast
import torch.nn.functional as F


def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()

    losses = AverageMeter()

    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    # for loop over one epoch
    for train_dict in bar:

        if scaler:
            with autocast(device_type=train_config.device):
                train_dict = {key: val.to(train_config.device) for key, val in train_dict.items()}
                # Forward pass
                features = model(train_dict)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss = loss_function(features, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features, model.logit_scale.exp())
                losses.update(loss.item())
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        else:

            # data (batches) to device
            train_dict = {key: val.to(train_config.device) for key, val in train_dict.items()}

            # Forward pass
            features = model(train_dict)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss = loss_function(features, model.module.logit_scale.exp())
            else:
                loss = loss_function(features, model.logit_scale.exp())
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        if train_config.verbose:

            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}

            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    model.eval()

    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    features_dict = {}
    labels_dict = {}

    with torch.no_grad():
        for test_dict, ids in bar:
            ids = ids.to(train_config.device)

            with autocast(device_type=train_config.device):
                # Move inputs to device
                test_dict = {key: val.to(train_config.device) for key, val in test_dict.items()}
                output_dict = model(test_dict)  # output_dict: {modality: feature_tensor}

                # Normalize each modality if required
                if train_config.normalize_features:
                    output_dict = {key: F.normalize(val, dim=-1) for key, val in output_dict.items()}

            # Accumulate features per modality
            for key, feat in output_dict.items():
                feat = feat.to(torch.float32)  # ensure fp32 for sim
                if key not in features_dict:
                    features_dict[key] = [feat]
                    labels_dict[key] = [ids]
                else:
                    features_dict[key].append(feat)
                    labels_dict[key].append(ids)

    # Concatenate features and labels per modality
    for key in features_dict:
        features_dict[key] = torch.cat(features_dict[key], dim=0)
        labels_dict[key] = torch.cat(labels_dict[key], dim=0)

    if train_config.verbose:
        bar.close()

    return features_dict, labels_dict
