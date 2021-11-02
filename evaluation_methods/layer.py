"""library module with functions to analyze and plot the activations of
   certain layers of a given model.
"""
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np
from architectures.cnn import CNN
from custom_modules.dictionaries import get_dicts
from custom_modules.dataset import TensorDataset
from custom_modules.tinyimagenet_utils import get_tinyimagenet_data

BCE, WASSERSTEIN, KLDIV, MinMax = 0, 1, 2, 3

def get_date() -> str:
    """
    helper function to generate a timestamp string
    :return: a printable timestamp string
    """

    date = datetime.now()
    img_year = "%04d" % (date.year)
    img_month = "%02d" % (date.month)
    img_date = "%02d" % (date.day)
    img_hour = "%02d" % (date.hour)
    img_mins = "%02d" % (date.minute)
    timestamp = "{}.{}.{} {}:{}".format(img_date, img_month, img_year, img_hour, img_mins)
    return timestamp

def get_loader(used_dataset: int) -> DataLoader:
    """
    helper function to initialize and provide two test data loaders.
    First one is unshuffled while the second one is shuffled.

    :param used_dataset: specifies which dataset should be provided by the dataloader.
                         0: CIFAR10, 1: CIFAR100, 2: TinyImageNet
    :return: two dataloaders with the testdata of the specified dataset. The first to search
             for the target image which should get missclassified and the shuffled one to
             randomly pick a image of the new_class for comparison.
    """
    #check if the cifar datasets have to be downloaded
    download_cifar10 = True
    download_cifar100 = True
    if os.path.isdir('./data/cifar-10-batches-py'):
        download_cifar10 = False
    if os.path.isdir('./data/cifar-100-python'):
        download_cifar100 = False
    if not os.path.isdir('./data/tiny-imagenet-200'):
        raise FileNotFoundError("Couldn't find TinyImageNet data at [./data/tiny-imagenet-200]")

    if used_dataset == 2:
        _, _, test_data, test_labels = get_tinyimagenet_data()
        dataset = TensorDataset(test_data, test_labels, transform=transforms.ToTensor())

    elif used_dataset == 1:
        dataset = CIFAR100(root='./data', train=False, download=download_cifar100,
                           transform=transforms.ToTensor())
    else:
        dataset = CIFAR10(root='./data', train=False, download=download_cifar10,
                          transform=transforms.ToTensor())

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    loader2 = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    return loader, loader2

def get_model(model_name: str, is_resnet: bool, used_dataset: int) -> nn.Sequential:
    """
    helper function to create and initialize two models. One complete model and
    one model without the last dense layer. Both models are used to plot the
    resulting activations.

    :param model_name: the path of the model which should get loaded
    :param is_resnet: specifies if a resnet architecture should be used
    :param used_dataset: specifies num classes of the model's output layer.
                         0: CIFAR10, 1: CIFAR100, 2: TinyImageNet
    :returns: the model with a modified classifier to show the activations.
    """
    if used_dataset == 2:
        num_classes = 200
    elif used_dataset == 1:
        num_classes = 100
    else:
        num_classes = 10

    if not is_resnet:
        net = CNN(num_classes=num_classes)
        net.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )

        net_complete = CNN(num_classes=num_classes)

    else:
        net = models.resnet18()
        net.fc = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )
        net_complete = models.resnet18()
        net_complete.fc = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )

    checkpoint = torch.load('./model_saves/'+str(model_name)+'/'+str(model_name),
                            map_location=lambda storage, loc: storage)
    checkpoint_complete = torch.load('./model_saves/'+str(model_name)+'/'+str(model_name),
                                     map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net'], strict=False)
    net_complete.load_state_dict(checkpoint_complete['net'], strict=False)
    net.eval()
    net_complete.eval()

    return net, net_complete

# ---------------------------------------------------

def analyze_layers(epsilon: float, pgd_iters: int, target_class: int, new_class: int,
                   save_path: str, model_name: str, pert_count: float, loss_fn: int,
                   device_name: str, layer_cut: int, used_dataset: int, is_resnet: bool,
                   target_id: int = None) -> None:
    """
    analyzes the whole layer activation of the penultimate and last layer and saves the images
    in the results folder.

    :param epsilon: the used epsilon for the PGD attack. Used only for visualization
    :param pgd_iters: the iterations of the PGD attack. Used only for visualization
    :param target_class: the target class for the image which should get missclassified
    :param new_class: the class as which the chosen image should get missclassified
    :param save_path: the path as which the resulting image should be saved
    :param model_name: the name of the model which should get analyzed
    :param pert_count: the percentage of the dataset perturbation. Used only for visualization
    :param loss_fn: the used loss function for the attack. Used only for visualization
    :param device_name: the device on which the analysis should be computed
    :param layer_cut: specifies how many layers of the model got cut off
    :param used_dataset: the used dataset for the attack 0:CIFAR10, 1:CIFAR100, 2:TinyImageNet
    :param is_resnet: specifies if the resnet architecture should be used
    :param target_id: the id of the image which should get missclassified after the attack

    :return: None
    """
    print("[ Initialize .. ]")
    global device
    device = device_name
    class_dict, _, _ = get_dicts(used_dataset)

    # generate several strings according to the chosen parameters
    # those strings are used to provide informations in the plots
    loss_function = ""
    if loss_fn == KLDIV:
        loss_function = "KLDiv"
    if loss_fn == WASSERSTEIN:
        loss_function = "Wasserstein"
    if loss_fn == BCE:
        loss_function = "BCE_WithLogits"
    if loss_fn == MinMax:
        loss_function = "MinMax Loss"

    layer_string = ""
    if layer_cut == 2:
        layer_string = "without 2 last layers"
    if layer_cut == 1:
        layer_string = "without 1 last layers"

    # obtain the timestamp, aswell as the models, dataset loaders and the
    # instance of the adversarial attack class
    date = get_date()
    model, model_complete = get_model(model_name, is_resnet, used_dataset)
    # both dataset loaders provide the cifar10 test samples, the first one is
    # deterministic while the second one is shuffled
    loader1, loader2 = get_loader(used_dataset)

    print("[ Analyze Layers .. ]")
    input_target, input_new_class = None, None

    # search for the image which should get misclassified (the first one of the class
    # if no image id is given)
    for batch_idx, (input, target) in enumerate(loader1):
        if target_id is not None:
            if batch_idx == target_id:
                input_target = input
        else:
            if target == target_class:
                input_target = input

    # and for the first image of the correct class in the shuffled dataset
    for _, (inputs2, targets2) in enumerate(loader2):
        if targets2 == new_class:
            input_new_class = inputs2

    # calculate the activations of the penultimate and last dense layer for the
    # image that should be misclassified
    activations = model(input_target)
    activations = np.reshape(activations.detach().numpy(), (16, 32))
    activations_last = model_complete(input_target).detach()

    # calculate the activations of the penultimate and last dense layer for a
    # original image from the new class
    activations2 = model(input_new_class)
    activations2 = np.reshape(activations2.detach().numpy(), (16, 32))
    activations_last2 = model_complete(input_new_class).detach()

    # generate softmaxed versions of the last dense layer activations
    softmaxed_last = F.softmax(activations_last, dim=1)
    strongest_class = softmaxed_last.max(-1)[1]
    softmaxed_last2 = F.softmax(activations_last2, dim=1)
    strongest_class2 = softmaxed_last2.max(-1)[1]


    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    fig.suptitle("model activations | input_id:{} | $\epsilon={}$ " \
                 "| iters={} | {} Perturbation | {} | {} | {} \n " \
                 " new_class: {} ({})".format(target_id, epsilon, pgd_iters, pert_count,
                                              loss_function, layer_string, date,
                                              new_class, class_dict[new_class]))

    # plots the input image which should get misclassified as the new class
    # and its activations for the last and penultimate dense layer
    axes[0][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[0][0].set_title("{} Input Image".format(class_dict[target_class]))
    axes[0][0].axis('off')
    axes[0][1].imshow(activations, cmap="cool")
    axes[0][1].set_title("Activations Penultimate Layer")
    axes[0][1].axis('off')
    axes[0][2].imshow(activations_last.numpy(), cmap="cool")
    axes[0][2].set_title("Activations Output Layer")
    axes[0][2].get_yaxis().set_visible(False)
    axes[0][2].get_xaxis().set_ticks(np.arange(len(class_dict)))
    axes[0][3].imshow(softmaxed_last.numpy(), cmap="cool")
    axes[0][3].set_title("Strongest Class: {} ({})".format(class_dict[int(strongest_class)],
                                                           int(strongest_class)))
    axes[0][3].get_yaxis().set_visible(False)
    axes[0][3].get_xaxis().set_ticks(np.arange(len(class_dict)))

    # plots the example image of the new class
    # and its activations for the last and penultimate dense layer
    axes[1][0].imshow(np.moveaxis(input_new_class.cpu().squeeze().numpy(), 0, -1))
    axes[1][0].set_title("Example {} Input Image".format(class_dict[new_class]))
    axes[1][0].axis('off')
    axes[1][1].imshow(activations2, cmap="cool")
    axes[1][1].set_title("Activations Penultimate Layer")
    axes[1][1].axis('off')
    axes[1][2].imshow(activations_last2.numpy(), cmap="cool")
    axes[1][2].set_title("Activations Output Layer")
    axes[1][2].get_yaxis().set_visible(False)
    axes[1][2].get_xaxis().set_ticks(np.arange(len(class_dict)))
    axes[1][3].imshow(softmaxed_last2.numpy(), cmap="cool")
    axes[1][3].set_title("Strongest Class: {} ({})".format(class_dict[int(strongest_class2)],
                                                           int(strongest_class2)))
    axes[1][3].get_yaxis().set_visible(False)
    axes[1][3].get_xaxis().set_ticks(np.arange(len(class_dict)))

    plt.savefig('./'+ str(save_path) +'/layer_eval_'+ str(model_name) +'.png', dpi=400)
    plt.close('all')
