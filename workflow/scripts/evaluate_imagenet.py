import argparse

import numpy as np
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.models import (ResNet18_Weights, ResNet34_Weights,
                                ResNet50_Weights, ResNet101_Weights,
                                ResNet152_Weights, resnet18, resnet34,
                                resnet50, resnet101, resnet152)
from tqdm import tqdm

models = {
    "resnet18": {"model": resnet18, "weights": ResNet18_Weights.IMAGENET1K_V1},
    "resnet34": {"model": resnet34, "weights": ResNet34_Weights.IMAGENET1K_V1},
    "resnet50": {"model": resnet50, "weights": ResNet50_Weights.IMAGENET1K_V2},
    "resnet101": {"model": resnet101, "weights": ResNet101_Weights.IMAGENET1K_V2},
    "resnet152": {"model": resnet152, "weights": ResNet152_Weights.IMAGENET1K_V2},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--out_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name

    model = models[model_name]["model"]
    weights = models[model_name]["weights"]

    model = model(weights=weights)
    model.cuda()
    model.eval()

    dataset = ImageNet(
        root="data/imagenet/raw_data/", split="val", transform=weights.transforms()
    )
    dl = DataLoader(dataset, batch_size=128, shuffle=False)
    n_batches = len(dl)

    all_probs = []
    for tensors in tqdm(dl, total=n_batches):
        with torch.no_grad():
            img, labels = tensors
            img_ = img.cuda()
            outs_ = model(img_)
            probs_ = softmax(outs_, dim=1)
            probs = probs_.cpu().detach().numpy()
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs)
    np.save(args.out_path, all_probs)
