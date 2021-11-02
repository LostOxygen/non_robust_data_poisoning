"""library module with functions to compute a tsne embedding for a given model and
   dataset
"""
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from custom_modules.dataset import TensorDataset
from custom_modules.tinyimagenet_utils import get_tinyimagenet_data
from custom_modules.dictionaries import get_dicts


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

def get_cmap(n, name='hsv'):
    """
    returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.

    :param n: index range on which rgb colors should be mapped on
    :param name: colormode of the cmap
    :return: matplotlib cmap function
    """
    return plt.cm.get_cmap(name, n)


def scale_to_01_range(x) -> np.ndarray:
    """
    scales the input to a range between 0 and 1

    :param x: unscaled input
    :return: scaled input
    """
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


def get_dataset(used_dataset: int) -> DataLoader:
    """
    helper function to create dataset loaders and the according class dictionary
    :param used_dataset: specifies which dataset should be provided by the dataloader.
                         0: CIFAR10, 1: CIFAR100, 2: TinyImageNet

    :return: train_dataset, train dataset loader and train dataset loader with the whole
             dataset as the batch size (used to copy the dataset efficient)
    """
    if used_dataset == 2:
        train_data, train_labels, _, _ = get_tinyimagenet_data()
        train_dataset = TensorDataset(train_data, train_labels, transform=transforms.ToTensor())

    elif used_dataset == 1:
        train_dataset = CIFAR100(root='./data', train=True, download=False,
                                 transform=transforms.ToTensor())
    else:
        train_dataset = CIFAR10(root='./data', train=True, download=False,
                                transform=transforms.ToTensor())
    dataset_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)
    whole_loader = DataLoader(train_dataset, batch_size=len(train_dataset),
                              shuffle=False, num_workers=4)
    return train_dataset, dataset_loader, whole_loader

def compute_tsne(model: nn.Sequential, data_suffix: str, dataset: int) -> None:
    """
    main function to compute the tsne embedding for a given model and dataset
    for detailed explanation see:
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    :param model: the model object which should be used for the TSNE
    :param data_suffix: specifies the name of the used model
    :param dataset: specifies which dataset should be used for the TSNE.
                        0: CIFAR10, 1: CIFAR100, 2: TinyImageNet

    :return: None
    """
    print("[ Initialize.. ]")
    train_dataset, train_loader, whole_loader = get_dataset(used_dataset=dataset)
    class_dict, _, _ = get_dicts(used_dataset=dataset)

    features = None
    for _, (input, _) in tqdm(enumerate(train_loader), desc="Running Model Inference"):
        input = input.to(DEVICE)

        with torch.no_grad():
            output = model.forward(input)

        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    print("[ Running TSNE ]")
    tsne = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features)
    t_x = tsne[:, 0]
    t_y = tsne[:, 1]

    t_x = scale_to_01_range(t_x)
    t_y = scale_to_01_range(t_y)

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("t_SNE | {}".format(data_suffix))
    a_x = fig.add_subplot(111)

    print("[ Visualize.. ]")
    classes = list(np.arange(0, len(class_dict)))
    cmap = get_cmap(len(classes))

    _, labels = list(whole_loader)[0]
    for single_class in classes:
        # labels = train_dataset.targets
        indices = [i for i, l in enumerate(labels) if l == single_class]

        current_tx = np.take(t_x, indices)
        current_ty = np.take(t_y, indices)

        a_x.scatter(current_tx, current_ty, cmap=cmap(single_class), label=class_dict[single_class])

    a_x.axis('off')
    a_x.legend(loc='best')
    plt.rcParams.update({'font.size': 16})

    result_path = 'results/{}_results/'.format(data_suffix)
    if not os.path.isdir(result_path):
        if not os.path.isdir('results/'):
            os.mkdir('results/')
        os.mkdir(result_path)

    plt.savefig(result_path + 'tsne_{}.png'.format(data_suffix), dpi=400)
    #plt.show()
