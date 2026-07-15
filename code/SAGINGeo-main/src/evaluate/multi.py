import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict
import itertools


def evaluate(config,
             model,
             dataloader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):

    print("\nExtract Features:")
    features_dict, labels_dict = predict(config, model, dataloader)

    print("Compute Cross-View Scores:")
    modal_names = list(features_dict.keys())

    results = {}

    # Compute all unique unordered modality pairs (no duplicates, no self-pairs)
    for mod1, mod2 in itertools.combinations(modal_names, 2):
        print(f"\nEvaluating {mod1} → {mod2}")
        r1 = calculate_scores(
            query_features=features_dict[mod1],
            reference_features=features_dict[mod2],
            query_labels=labels_dict[mod1],
            reference_labels=labels_dict[mod2],
            step_size=step_size,
            ranks=ranks
        )
        results[f"{mod1}->{mod2}"] = r1

        print(f"\nEvaluating {mod2} → {mod1}")
        r2 = calculate_scores(
            query_features=features_dict[mod2],
            reference_features=features_dict[mod1],
            query_labels=labels_dict[mod2],
            reference_labels=labels_dict[mod1],
            step_size=step_size,
            ranks=ranks
        )
        results[f"{mod2}->{mod1}"] = r2


    return results[list(results.keys())[0]], features_dict


def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1, 5, 10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)

    steps = Q // step_size + 1

    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()

    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i

    similarity = []

    for i in range(steps):
        start = step_size * i
        end = min(start + step_size, Q)
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())

    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk.append(R // 100)

    results = np.zeros([len(topk)])

    bar = tqdm(range(Q))

    for i in bar:
        # If ground-truth label doesn't exist in reference set, skip
        if query_labels_np[i] not in ref2index:
            continue

        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        higher_sim = similarity[i, :] > gt_sim
        ranking = higher_sim.sum().item()

        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.

    results = results / Q * 100.

    bar.close()
    time.sleep(0.1)

    string = []
    for i in range(len(topk) - 1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
    string.append('Recall@top1: {:.4f}'.format(results[-1]))
    print(' - '.join(string))

    return results[0]


