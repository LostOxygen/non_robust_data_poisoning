"""
library module with functions to check the attack success and success ratios
"""
from typing import Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from architectures.cnn import CNN
from custom_modules.dataset import TensorDataset
from custom_modules.tinyimagenet_utils import get_tinyimagenet_data


def get_loader(used_dataset: int) -> DataLoader:
    """
    helper function to initialize and provide a test data loader.

    :param used_dataset: specifies which dataset should be provided by the dataloader.
                         0: CIFAR10, 1: CIFAR100, 2: TinyImageNet
    :return: a dataloader with the testdata of the specified dataset.
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
    return loader


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
        net_complete = CNN(num_classes=num_classes)

    else:
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

    checkpoint_complete = torch.load('./model_saves/'+str(model_name)+'/'+str(model_name),
                                     map_location=lambda storage, loc: storage)
    net_complete.load_state_dict(checkpoint_complete['net'], strict=False)
    net_complete.eval()
    return net_complete


def check_success(new_class: int, target_id: int, is_resnet: bool, model_name: str,
                  used_dataset: int, untargeted: bool) -> bool:
    """
    Analyzes the whole layer activation of the penultimate and last layer and saves the images
    in the results folder.

    :param new_class: the class as which the chosen image should get missclassified
    :param model_name: the name of the model which should get analyzed
    :param device_name: the device on which the analysis should be computed
    :param used_dataset: the used dataset for the attack 0:CIFAR10, 1:CIFAR100, 2:TinyImageNet
    :param is_resnet: specifies if the resnet architecture should be used
    :param target_id: the id of the image which should get missclassified after the attack
    :param untargeted: specifies if the attack is untargeted

    :return: True if the attack succeeded and False if not
    """
    print("[ Check if the attack succeeded: ]")
    model = get_model(model_name, is_resnet, used_dataset)
    test_loader = get_loader(used_dataset)

    # search for the image which should get misclassified due to the attack
    prediction = None
    target_label = None
    for batch_idx, (input, target) in enumerate(test_loader):
        if batch_idx == target_id:
            target_label = target
            prediction = F.softmax(model(input).detach(), dim=-1).max(-1)[1]

    # for an untargeted attack, the prediction simply has to be wrong instead of a specific class
    if untargeted:
        success_flag = prediction != target_label
    else:
        success_flag = prediction == new_class

    if success_flag:
        print("[ Attack successful! ]")
    else:
        print("[ Attack unsuccessful .. ]")

    return success_flag


def save_dataframe(epsilon: float, layer_cuts: int, target_class: str, new_class: str,
                   loss_fn: int, pert_count: float, current_total_attacks: int,
                   acc_single: list, successful_attacks: int, acc_whole: float,
                   rand_img: bool, best_img: bool, prefix: str, num_clusters: int) -> None:
    """
    Saves the parameters and accuracies of an successful attack as a dataframe and excel sheet

    :param epsilon: used epsilon constraint for the perturbation
    :param layer_cuts: number of layer which got cut off the model
    :param target_class: string of the target class name
    :param new_class: string of the new class name
    :param loss_fn: used loss function for the attack
    :param pert_count: how much of the poison class got perturbated
    :param total_attacks: Number of total attacks
    :param acc_single: List with the accuracy of the target and the new class
    :param successful_attacks: Number of successful attacks
    :param acc_whole: test accuracy of the whole model
    :param rand_img: flag that indicates if a random image instead of the most suitable one is used
                     -> will create a extra row for saving its parameters
    :param prefix: the model name prefix to seperate databases for models and datasets
    :param num_clusters: number of clusters used for an untargeted attack

    :return: None (but saves data as a dataframe)
    """
    print("[ Creating DataFrame ]")
    current_const = "{} to {}{}{}".format(target_class, new_class, " rand" if rand_img else "",
                                          " best" if best_img else "")

    total_attacks_list = list([current_total_attacks])
    successful_attacks_list = list([successful_attacks])
    success_ratio_list = list([np.round(successful_attacks / current_total_attacks, 2)])
    target_accuracy_list = list([acc_single[0]])
    mean_target_accuracy_list = list([acc_single[0]])
    poison_accuracy_list = list([acc_single[1]])
    mean_poison_accuracy_list = list([acc_single[0]])
    test_accuracy_list = list([acc_whole])
    mean_accuracy_list = list([acc_whole])
    num_clusters_list = list([num_clusters])

    epsilon_list = list([epsilon])
    loss_fn_list = list([loss_fn])
    pert_count_list = list([pert_count])
    layer_cuts_list = list([layer_cuts])

    new_data_frame = None
    if os.path.isfile("results/{}_attack_results.pkl".format(prefix)):
        # load the existing data
        old_data_frame = pd.read_pickle("results/{}_attack_results.pkl".format(prefix))
        # select the row of the current class constellation, but only if the const. exists
        if current_const in old_data_frame.constellation.values:
            # case where a dataframe with the same class combination already exists
            # -> fetch the old data and append the new data
            old_data = old_data_frame[old_data_frame["constellation"] == current_const]
            # also fetch every row which is NOT the current constellation
            remaining_data = old_data_frame[old_data_frame["constellation"] != current_const]

            # overwrite and append new data to the old data
            total_attacks_list = list([current_total_attacks])
            successful_attacks_list = list([successful_attacks])
            success_ratio_list = list([np.round(successful_attacks / current_total_attacks, 2)])
            target_accuracy_list = list(old_data["target_accuracy"])[0]
            target_accuracy_list.append(acc_single[0])
            poison_accuracy_list = list(old_data["poison_accuracy"])[0]
            poison_accuracy_list.append(acc_single[1])
            test_accuracy_list = list(old_data["test_accuracy"])[0]
            test_accuracy_list.append(acc_whole)
            num_clusters_list = list(old_data["clusters"])[0]
            num_clusters_list.append(num_clusters)

            mean_accuracy_list = np.round(sum(\
                test_accuracy_list)/len(test_accuracy_list), 2)
            mean_accuracy_list = np.array(mean_accuracy_list).tolist()
            mean_target_accuracy_list = np.round(sum(\
                target_accuracy_list)/len(target_accuracy_list), 2)
            mean_target_accuracy_list = np.array(mean_target_accuracy_list).tolist()
            mean_poison_accuracy_list = np.round(sum(\
                poison_accuracy_list)/len(poison_accuracy_list), 2)
            mean_poison_accuracy_list = np.array(mean_poison_accuracy_list).tolist()

            epsilon_list = list(old_data["epsilon"])[0]
            epsilon_list.append(epsilon)
            loss_fn_list = list(old_data["loss_fn"])[0]
            loss_fn_list.append(loss_fn)
            pert_count_list = list(old_data["pert_count"])[0]
            pert_count_list.append(pert_count)
            layer_cuts_list = list(old_data["layer_cuts"])[0]
            layer_cuts_list.append(layer_cuts)

            # create dictionary to save the accuracies and results as a dataframe
            # for each constellation
            result_data = {"constellation": current_const,
                           "successful_attacks": successful_attacks_list[0],
                           "total_attacks": total_attacks_list[0],
                           "success_ratio": success_ratio_list[0],
                           "target_accuracy": target_accuracy_list,
                           "mean_target_accuracy": mean_target_accuracy_list,
                           "poison_accuracy":poison_accuracy_list,
                           "mean_poison_accuracy": mean_poison_accuracy_list,
                           "test_accuracy": test_accuracy_list,
                           "mean_accuracy": mean_accuracy_list,
                           "epsilon": epsilon_list,
                           "loss_fn": loss_fn_list,
                           "pert_count": pert_count_list,
                           "layer_cuts": layer_cuts_list,
                           "clusters": num_clusters_list
                           }

            # append the updated row back to the remaining data
            new_data_frame = remaining_data.append(result_data, ignore_index=True)

        else:
            # case where a dataframe already exists, but the class constellation is new
            # -> create dictionary and append it as a row below the existing class constellations
            # create dictionary to save the accuracies and results as a dataframe
            # for each constellation
            result_data = {"constellation": current_const,
                           "successful_attacks": successful_attacks_list[0],
                           "total_attacks": total_attacks_list[0],
                           "success_ratio": success_ratio_list[0],
                           "target_accuracy": target_accuracy_list,
                           "mean_target_accuracy": mean_target_accuracy_list[0],
                           "poison_accuracy": poison_accuracy_list,
                           "mean_poison_accuracy": mean_poison_accuracy_list[0],
                           "test_accuracy": test_accuracy_list,
                           "mean_accuracy": mean_accuracy_list[0],
                           "epsilon": epsilon_list,
                           "loss_fn": loss_fn_list,
                           "pert_count": pert_count_list,
                           "layer_cuts": layer_cuts_list,
                           "clusters": num_clusters_list
                           }

            new_data_frame = old_data_frame.append(result_data, ignore_index=True)

    else:
        # normale case, where no saved data frame exists:
        # create dictionary to save the accuracies and results as a dataframe
        # for each constellation
        result_data = {"constellation": [current_const],
                       "successful_attacks": successful_attacks_list,
                       "total_attacks": total_attacks_list,
                       "success_ratio": success_ratio_list,
                       "target_accuracy": [target_accuracy_list],
                       "mean_target_accuracy": mean_target_accuracy_list,
                       "poison_accuracy": [poison_accuracy_list],
                       "mean_poison_accuracy": mean_poison_accuracy_list,
                       "test_accuracy": [test_accuracy_list],
                       "mean_accuracy": mean_accuracy_list,
                       "epsilon": [epsilon_list],
                       "loss_fn": [loss_fn_list],
                       "pert_count": [pert_count_list],
                       "layer_cuts": [layer_cuts_list],
                       "clusters": [num_clusters_list]
                       }

        new_data_frame = pd.DataFrame(result_data)

    try:
        new_data_frame.to_pickle("results/{}_attack_results.pkl".format(prefix))
        new_data_frame.to_excel("results/{}_attack_results.xls".format(prefix))
        print("[ Saved DataFrame ]")
        return 0
    except RuntimeError as err:
        print("Could not save DataFrame..")
        print(err)


def get_best_parameters(target_class: str, new_class: str, prefix: str) -> Union[list, list, list, list]:
    """
    Loads the existing attack results and returns the successful parameters for a given class
    constellation.
    :param prefix: the model name prefix to seperate databases for models and datasets
    :param target_class: string of the target class name
    :param new_class: string of the new class name

    :returns: Union of epsilon, loss function, perturbation count and layer cuts
    """
    print("[ Loading Successful Attack Parameters ]")
    current_const = "{} to {}".format(target_class, new_class)

    if os.path.isfile("results/{}_attack_results.pkl".format(prefix)):
        # load the existing data
        old_data_frame = pd.read_pickle("results/{}_attack_results.pkl".format(prefix))
        # select the row of the current class constellation, but only if the column exists
        if current_const in old_data_frame.constellation.values:
            # case where a dataframe with the same class combination already exists
            # -> fetch the old data and append the new data
            old_data = old_data_frame[old_data_frame["constellation"] == current_const]
            epsilons = list(old_data["epsilon"])[0]
            loss_fns = list(old_data["loss_fn"])[0]
            pert_counts = list(old_data["pert_count"])[0]
            layer_cuts = list(old_data["layer_cuts"])[0]

            # remove duplicates from the parameter lists
            epsilons = list(set(epsilons))
            loss_fns = list(set(loss_fns))
            pert_counts = list(set(pert_counts))
            layer_cuts = list(set(layer_cuts))

        else:
            raise RuntimeError("dataframe does not contain current class constellation")
    else:
        raise FileNotFoundError("attack_results dataframe not found")

    return epsilons, loss_fns, pert_counts, layer_cuts

def plot_attack_results(prefix: str) -> None:
    """
    Load the saved database with all previous successful attack parameters and its accuracies and so
    on to plot the success ratio for every class constellation
    :param prefix: the model name prefix to seperate databases for models and datasets

    :returns: None
    """
    print("[ Loading Successful Attack Parameters ]")
    if os.path.isfile("results/{}_attack_results.pkl".format(prefix)):
        # load the existing data
        data_frame = pd.read_pickle("results/{}_attack_results.pkl".format(prefix))
    else:
        print("{}_attack_results.pkl dataframe not found".format(prefix))
        return -1

    const_list = list()
    succ_ratio_list = list()
    mean_poison_accuracy = list()
    mean_target_accuracy = list()
    mean_accuracy = list()

    for constellation in data_frame.constellation.values:
        # iterate over all constellation in the dataframe
        const_list.append(constellation)
        # fetch data for current class constellation
        const_data = data_frame[data_frame["constellation"] == constellation]
        # append the success ratio
        succ_ratio_list.append(list(const_data["success_ratio"])[0])
        # build lists with the mean accuracies
        mean_poison_accuracy.append(list(const_data["mean_poison_accuracy"])[0])
        mean_target_accuracy.append(list(const_data["mean_target_accuracy"])[0])
        mean_accuracy.append(list(const_data["mean_accuracy"])[0])

    # ------------- plot and save the success ratio ----------
    fig, axis = plt.subplots(1, 1, figsize=(25, 7))
    rects1 = axis.bar(const_list, succ_ratio_list, 0.3)
    #axis.bar_label(rect1, padding=3)
    for rect in rects1:
        height = rect.get_height()
        axis.annotate('{}'.format(np.round(height, 2)),
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')
    axis.set_ylabel('Success Ratio')
    axis.set_title('Class Combinations: {}'.format(prefix))
    fig.tight_layout()
    plt.savefig('results/{}_success_ratio.png'.format(prefix))
    plt.close('all')

    # ------------- plot and save the accuracies ----------
    x_val = np.arange(len(const_list))
    width = 0.25
    fig, axis = plt.subplots(1, 1, figsize=(25, 7))

    rects1 = axis.bar(x_val - width, mean_accuracy, width, label="Complete Test",
                      color="cornflowerblue")
    rects2 = axis.bar(x_val, mean_target_accuracy, width, label="Target Test",
                      color="forestgreen")
    rects3 = axis.bar(x_val + width, mean_poison_accuracy, width, label="Poison Test",
                      color="firebrick")
    # axis.bar_label(rects1, padding=3)
    # axis.bar_label(rects2, padding=3)
    # axis.bar_label(rects3, padding=3)
    for rect in rects1:
        height = rect.get_height()
        axis.annotate('{}'.format(np.round(height, 2)),
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')

    for rect in rects2:
        height = rect.get_height()
        axis.annotate('{}'.format(np.round(height, 2)),
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')

    for rect in rects3:
        height = rect.get_height()
        axis.annotate('{}'.format(np.round(height, 2)),
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')

    axis.set_ylabel('Accuracy')
    axis.set_title('Class Combinations: {}'.format(prefix))
    axis.legend()
    axis.set_xticks(x_val)
    axis.set_xticklabels(const_list)

    fig.tight_layout()
    plt.savefig('results/{}_success_accuracies.png'.format(prefix))
    plt.close('all')
