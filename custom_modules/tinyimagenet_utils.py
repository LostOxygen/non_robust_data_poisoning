"""modified library for loading and using the TinyImageNet dataset
   from: https://colab.research.google.com/github/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/Train_ResNet_On_Tiny_ImageNet.ipynb#scrollTo=7TUH7bu7n5ta
"""
from imageio import imread
import torch
import numpy as np
from tqdm import tqdm

PATH = './data/tiny-imagenet-200/'

def get_id_dictionary() -> dict:
    """
    helper function which returns an dictionary with all ids of the TinyImageNet dataset

    :return: dictionary with the IDs mapped to their class labels
    """
    id_dict = {}
    for i, line in enumerate(open(PATH + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict

def get_class_to_id_dict() -> dict:
    """
    helper function to load tinyimagenet data and return it as training and test data

    :return: dictionary which maps the classes to their TinyImageNet ID
    """
    id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for _, line in enumerate(open(PATH + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        word = word.split(",")[0].lower()
        all_classes[n_id] = word.rstrip("\n")
    for key, value in id_dict.items():
        result[value] = (all_classes[key])

    return result

def get_tinyimagenet_data() -> np.ndarray:
    """
    helper function to load tinyimagenet data and return it as training and test data

    :return: traindata, trainlabels, testdata and testlabels of the TinyImageNet
    """
    print('[ Loading TinyImageNet ]')
    id_dict = get_id_dictionary()
    # preallocate data tensors
    train_data, test_data = torch.empty((100000, 64, 64, 3)), torch.empty((10000, 64, 64, 3))
    train_labels, test_labels = torch.empty((100000)), torch.empty((10000))
    j = 0
    for key, value in tqdm(id_dict.items()):
        for i in range(500):
            train_data[j+i] = torch.FloatTensor(imread(PATH + 'train/{}/images/{}_{}.JPEG'.format(
                key, key, str(i)), pilmode="RGB", as_gray=False))
            train_labels[j+i] = value
        j += 500

    k = 0
    for line in open(PATH + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        test_data[k] = torch.FloatTensor(imread(PATH + 'val/images/{}'.format(img_name),
                                                pilmode="RGB", as_gray=False))
        test_labels[k] = id_dict[class_id]
        k += 1

    # normalize the data
    train_data /= 255.
    test_data /= 255.

    # move channel dimension -> (Channel, Height, Width)
    train_data = torch.movedim(train_data, -1, -3)
    test_data = torch.movedim(test_data, -1, -3)

    # convert to LongTensors
    train_labels = train_labels.type(torch.LongTensor)
    test_labels = test_labels.type(torch.LongTensor)

    return train_data, train_labels, test_data, test_labels
