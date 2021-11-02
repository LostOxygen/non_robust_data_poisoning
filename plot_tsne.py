"""main file to compute and plot TSNE embeddings"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time
import argparse
import numpy as np
import torch
from tools.tsne import compute_tsne
from architectures.cnn import CNN
from architectures.resnet_tinyimagenet import tinyimg_resnet
from architectures.resnet_cifar import cifar_resnet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model(model_name: str, is_resnet: bool, used_dataset: int):
    """helper function to create and initialize the model."""
    if used_dataset == 2:
        num_classes = 200
    elif used_dataset == 1:
        num_classes = 100
    else:
        num_classes = 10

    if is_resnet:
        if used_dataset == 2:
            net = tinyimg_resnet(num_classes=num_classes)
        else:
            net = cifar_resnet(num_classes=num_classes)
    else:
        net = CNN(num_classes=num_classes)
    net = net.to(DEVICE)
    checkpoint = torch.load('./model_saves/'+str(model_name)+'/'+str(model_name),
                            map_location=torch.device(DEVICE))
    net.load_state_dict(checkpoint['net'])
    return net


def get_model_name(resnet: bool, dataset: int) -> str:
    """helper function to build the model string"""
    if dataset == 2:
        if resnet:
            model_name = 'resnet_tinyimagenet'
        else:
            model_name = 'basic_tinyimagenet'
    elif dataset == 1:
        if resnet:
            model_name = 'resnet_cifar100'
        else:
            model_name = 'basic_cifar100'
    else:
        if resnet:
            model_name = 'resnet_cifar10'
        else:
            model_name = 'basic_cifar10'
    return model_name


def main(mult: bool, resnet: bool, dataset: int) -> None:
    """main function to compute and plot TSNE embeddings"""
    start = time.perf_counter()

    model_name = get_model_name(resnet, dataset)
    print("Calculating TSNE embedding for: [{}]".format(model_name))
    model = get_model(model_name, resnet, dataset)
    compute_tsne(model, model_name, dataset)

    print("finished TSNE computation for [ {} ]".format(model_name))
    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"Computation time: {duration:0.4f} hours")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mult", "-m", help="Multiple TSNE", action='store_true')
    parser.add_argument("--resnet", "-r", help="uses resnet instead of the normal cnn",
                        action='store_true', default=False, required=False)
    parser.add_argument("--dataset", "-d", help="specifies the used dataset",
                        type=int, default=1, required=False)

    args = parser.parse_args()
    main(**vars(args))
