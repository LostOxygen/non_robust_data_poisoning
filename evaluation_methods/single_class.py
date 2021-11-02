"""library module with functions to analyze and plot the model accuracy for a specific
   class.
"""
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from architectures.cnn import CNN
from architectures.resnet_cifar import cifar_resnet
from architectures.resnet_tinyimagenet import tinyimg_resnet
from custom_modules.dictionaries import get_dicts
from custom_modules.dataset import TensorDataset
from custom_modules.tinyimagenet_utils import get_tinyimagenet_data
mpl.style.use('seaborn-deep')

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
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
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
    net = net.to(DEVICE)
    checkpoint = torch.load('./model_saves/'+str(model_name)+'/'+str(model_name),
                            map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    return net

def test(class_to_check: int, model_name: str, used_dataset: int,
         is_resnet: bool) -> np.ndarray:
    """
    helper function that iterates over the whole testset and sums the predictions
    of the model for every image that corresponds to the chosen class. Returns an array
    with the number of predictions for every class. -> to determine how the predictions
    for a manipulated class change after the attack

    :param class_to_check: the class which should be analyzed
    :param model_name: name/path of the model which should be used
    :param used_dataset: specifies num classes of the model's output layer.
                         0: CIFAR10, 1: CIFAR100, 2: TinyImageNet
    :param is_resnet: specifies if a resnet architecture should be used

    :return: a numpy array with the accuracy results for the specified class
    """
    net = get_model(model_name, is_resnet, used_dataset)
    test_loader = get_loader(used_dataset)
    #total number of predictions and number how often a certain class got predicted
    total = 0
    if used_dataset == 2:
        class_results = [0] * 200
    elif used_dataset == 1:
        class_results = [0] * 100
    else:
        class_results = [0] * 10

    # iterate over the whole test dataset and copy the data onto the DEVICE
    for _, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # check where the labels are the same as the class which should get checked
        # iterate only over the data points with the corresponding label
        target_ids = torch.where(targets == class_to_check)[0]
        for target_id in target_ids:
            output = net(torch.unsqueeze(inputs[target_id], 0))
            _, predicted = output.max(1)

            class_results[predicted] += 1
            total += 1

    # iterate over the results and convert into percent
    for id_i, item in enumerate(class_results):
        class_results[id_i] = (item / total)*100.
    del net
    return class_results


def evaluate_single_class(model_name: str, save_path: str, target_class: int, new_class: int,
                          epsilon: float, pgd_iters: int, pert_count: float, loss_function: int,
                          device_name: str, used_dataset: int, layer_cut: int,
                          is_resnet: bool) -> list:
    """
    evaluates the performance of the model for target and new class and visualize
    the results as a plot. Helps to understand how the overall model performance
    is affected by the training on the manipulated dataset and if the accuracy of
    the manipulated class changes. Most parameters are only used
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

    :return: the accuracy of both the target class and the new class
    """
    print('\n[ Evaluate Target/New Class Accuracy .. ]')
    global DEVICE
    DEVICE = device_name
    class_dict, _, loss_dict = get_dicts(used_dataset=used_dataset)

    # calculate the accuracy for the new and target class
    new_class_result = test(class_to_check=new_class, model_name=model_name,
                            used_dataset=used_dataset, is_resnet=is_resnet)
    target_class_result = test(class_to_check=target_class, model_name=model_name,
                               used_dataset=used_dataset, is_resnet=is_resnet)

    # create lists and strings with layer and label description
    layer_string = "without {} last layers".format(layer_cut)

    labels = list(class_dict.values())
    x = np.arange(len(class_dict))
    y = np.arange(100)
    width = 0.3

    # create plot and title
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("{} to {} | $\epsilon={}$ | iters={} | {} Perturbation | {} | {} | {}".format(
        class_dict[target_class], class_dict[new_class], epsilon, pgd_iters,
        pert_count, loss_dict[loss_function], layer_string, get_date()))


    # plot the accuracy of the class as which the target image should get misclassified
    std_rect = ax.bar(x - width/2, new_class_result, width, label='Acc.')
    ax.set_ylabel('Accuracy')
    ax.set_title('input: ' + str(class_dict[new_class]))
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.legend()
    for rect in std_rect:
        height = rect.get_height()
        ax.annotate('{}'.format(np.round(height, 1)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    # plot the accuracy of the target class
    std_rect2 = ax2.bar(x - width/2, target_class_result, width, label='Acc.')
    ax2.set_ylabel('')
    ax2.set_title('input: ' + str(class_dict[target_class]))
    ax2.set_xticks(x)
    ax2.set_yticks(y)
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels([])
    ax2.legend()
    for rect in std_rect2:
        height = rect.get_height()
        ax2.annotate('{}'.format(np.round(height, 1)),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.savefig('./'+ str(save_path) +'/evaluation_'+ str(model_name) +'.png', dpi=400)
    plt.close('all')
    # return the accuracy of the target and new class
    target_acc = np.round(target_class_result[target_class], 2)
    new_class_acc = np.round(new_class_result[new_class], 2)
    return [target_acc, new_class_acc]
