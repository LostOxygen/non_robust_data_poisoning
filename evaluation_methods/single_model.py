"""library module with functions to analyze and plot the complete and adversarial
   accuracy for a given model.
"""
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from advertorch.attacks import L2PGDAttack
from architectures.cnn import CNN
from architectures.resnet_cifar import cifar_resnet
from architectures.resnet_tinyimagenet import tinyimg_resnet
from custom_modules.dictionaries import get_dicts
from custom_modules.dataset import TensorDataset
from custom_modules.tinyimagenet_utils import get_tinyimagenet_data


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
    helper function to initialize and provide a testset data loader.

    :param used_dataset: specifies which dataset should be provided by the dataloader.
                         0: CIFAR10, 1: CIFAR100, 2: TinyImageNet
    :return: a testset dataloader with the specified dataset
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
        test_dataset = TensorDataset(test_data, test_labels, transform=transforms.ToTensor())

    elif used_dataset == 1:
        test_dataset = CIFAR100(root='./data', train=False, download=download_cifar100,
                                transform=transforms.ToTensor())
    else:
        test_dataset = CIFAR10(root='./data', train=False, download=download_cifar10,
                               transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)
    return test_loader


def get_model(model_name: str, is_resnet: bool, used_dataset: int) -> nn.Sequential:
    """
    helper function to create and initialize the model.
    :param model_name: the name/path of the model which should get loaded
    :param is_resnet: specifies if a resnet architecture should be used
    :param used_dataset: specifies num classes of the model's output layer.
                         0: CIFAR10, 1: CIFAR100, 2: TinyImageNet
    :return: returns the loaded model
    """
    if used_dataset == 2:
        num_classes = 200
    elif used_dataset == 1:
        num_classes = 100
    else:
        num_classes = 10

    if is_resnet:
        net = models.resnet18()
        net.fc = nn.Sequential(
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
    else:
        net = CNN(num_classes=num_classes)
    net = net.to(device)
    checkpoint = torch.load('./model_saves/'+str(model_name)+'/'+str(model_name),
                            map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    return net


def evaluate_single_model(model_name: str, save_path: str, target_class: int, new_class: int,
                          epsilon: float, pgd_iters: int, pert_count: float, loss_function: int,
                          device_name: str, used_dataset: int, layer_cut: int,
                          is_resnet: bool) -> float:
    """
    Evaluates the normal and adversarial model accuracy using unmodified
    and pertubed images of the whole specified test dataset. Most parameters are only used
    to provide information in the resulting image.

    :param model_name: name/path of the model which should get evalutated
    :param save_path: path where the resulting image should be saved
    :param target_class: the target class for the image which should get missclassified
    :param new_class: the class as which the chosen image should get missclassified
    :param epsilon: specifies the epsilon used for the PGD attack
    :param pgd_iters: specifies the iterations of the PGD attack
    :param pert_count: specifies how much of the dataset got perturbed
    :param loss_function: specifies the loss function which got used for the attack
    :param device_name: sets the device on which the computation should happen
    :param used_dataset: specifies which dataset got used for the attack.
                         0: CIFAR10, 1: CIFAR100, 2: TinyImageNet
    :param layer_cut: specifies how many layers of the classifier got cut off
    :param is_resnet: specifies if a resnet architecture should be used

    :return: the model test accuracy
    """
    global device
    device = device_name
    # get dictionaries
    class_dict, _, loss_dict = get_dicts(used_dataset)

    net = get_model(model_name, is_resnet, used_dataset)
    test_loader = get_loader(used_dataset)
    adversary = L2PGDAttack(net, loss_fn=nn.CrossEntropyLoss(), eps=0.25,
                            nb_iter=10, eps_iter=0.01, rand_init=True,
                            clip_min=0.0, clip_max=1.0, targeted=False)

    print('\n[ Evaluate Model Accuracy .. ]')
    net.eval()

    net_benign_correct = 0
    net_adv_correct = 0
    total = 0

    # iterate over the whole test set and copy data to DEVICE
    # create an adversary perturbed version for every image
    # predict both the benign and the adversary image and check if
    # the prediction is correct
    for _, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = adversary.perturb(inputs, targets)

        outputs = net(inputs)
        _, predicted = outputs.max(1)
        net_benign_correct += predicted.eq(targets).sum().item()

        adv_outputs = net(adv)
        _, predicted = adv_outputs.max(1)
        net_adv_correct += predicted.eq(targets).sum().item()


    # convert to percent and create a list to plot
    net_benign_correct = (net_benign_correct / total)*100.
    net_adv_correct = (net_adv_correct / total)*100.

    benign = [net_benign_correct]
    advs = [net_adv_correct]

    labels = [model_name]

    # visualize the lists of the benign and adversary prediction accuracy
    x = np.arange(len(labels))
    width = 0.03
    fig, ax = plt.subplots(figsize=(15, 5))
    # create to rectangles for benign and adversary accuracy
    std_rect = ax.bar(x - width/2, benign, width, label='Std. Acc.')
    advs_rect = ax.bar(x + width/2, advs, width, label='Advs. Acc.')
    layer_string = ""
    if layer_cut == 2:
        layer_string = "without 2 last layers"
    if layer_cut == 1:
        layer_string = "without 1 last layers"

    ax.set_ylabel('Accuracy in Percent')
    ax.set_title("model accuracy: {} to {} | $\epsilon={}$ | iters={} | {} Perturbation | {} | {} | {}".format(
        class_dict[target_class], class_dict[new_class], epsilon,
        pgd_iters, pert_count, loss_dict[loss_function], layer_string, get_date()))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # plot the benign accuracy
    for rect in std_rect:
        height = rect.get_height()
        ax.annotate('{}'.format(np.round(height, 1)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    # plot the adversary accuracy
    for rect in advs_rect:
        height = rect.get_height()
        ax.annotate('{}'.format(np.round(height, 1)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    fig.tight_layout()
    plt.savefig('./'+ str(save_path) +'/model_acc_'+ str(model_name) +'.png', dpi=400)
    plt.close('all')

    return np.round(net_benign_correct, 2)
