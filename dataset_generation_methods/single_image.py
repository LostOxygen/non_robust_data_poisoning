"""library module to create perturbed dataset which leads
   to a misclassification of one specific image when trained on
"""
import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.cluster import KMeans
from advertorch.attacks import L2PGDAttack
from architectures.cnn import CNN
from custom_modules.loss import WassersteinLoss, KLDivLoss, MinMaxLoss
from custom_modules.dataset import TensorDataset
from custom_modules.tinyimagenet_utils import get_tinyimagenet_data

BCE, WASSERSTEIN, KLDIV, MinMax = 0, 1, 2, 3

def get_model(model_name: str, device_string: str, is_resnet: bool,
              used_dataset: int, layers: int) -> nn.Sequential:
    """
    helper function to cut the last N dense layers. If layers==None the normal complete
    model is returned instead.

    :param model_name: the model name/path which should get loaded
    :param device_string: the device on which the model should be computed on
    :param is_resnet: flag to use the resnet architecture instead of the CNN
    :param used_dataset: specifies num classes of the model's output layer.
                         0: CIFAR10, 1: CIFAR100, 2: TinyImageNet
    :param layers: specifies how many layers of the classifier should get cut off

    :return: the specified model with a modified classifier to obtain the activations
    """
    if used_dataset == 2:
        num_classes = 200
    elif used_dataset == 1:
        num_classes = 100
    else:
        num_classes = 10

    if not is_resnet:
        net = CNN(num_classes=num_classes)
        if layers == 2:
            net.fc = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(4096, 1024),
                nn.ReLU(inplace=True)
            )
        if layers == 1:
            net.fc = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(4096, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True)
            )
        if layers is None:
            net.fc = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(4096, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)
            )
        # replace the whole
        if layers == -1:
            net.fc = nn.Sequential(
                nn.Identity()
            )
    else:
        net = models.resnet18()
        if layers == 2:
            net.fc = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(4096, 1024),
                nn.ReLU(inplace=True)
            )
        elif layers == 1:
            net.fc = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(4096, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True)
            )
        elif layers is None:
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
        # replace the whole
        elif layers == -1:
            net.fc = nn.Sequential(
                nn.Identity()
            )


    checkpoint = torch.load("./model_saves/{0}/{1}".format(model_name, model_name))
    net.load_state_dict(checkpoint['net'], strict=False)
    net = net.to(device_string)
    net.eval()
    return net


def get_data(used_dataset: int) -> torch.utils.data.DataLoader:
    """
    helper function to initialize and provide a trainset loader as well as the
    normal train and test sets to perform copy operations on them

    :param used_dataset: specifies which dataset should be provided by the dataloader.
                         0: CIFAR10, 1: CIFAR100, 2: TinyImageNet
    :return: a dataloader for the train/test data and both the raw train and test data which is
             used  for copy operations to preallocate unmodified dataset parts and stuff.
    """
    #check if the cifar datasets have to be downloaded
    download_cifar10 = True
    download_cifar100 = True
    if os.path.isdir('./data/cifar-10-batches-py'):
        download_cifar10 = False
    if os.path.isdir('./data/cifar-100-python'):
        download_cifar100 = False
    if not os.path.isdir('./data/tiny-imagenet-200'):
        raise FileNotFoundError("Could not find TinyImageNet data at [./data/tiny-imagenet-200]")

    if used_dataset == 2:
        train_data, train_labels, test_data, test_labels = get_tinyimagenet_data()
        train_dataset = TensorDataset(train_data, train_labels, transform=transforms.ToTensor())
        test_dataset = TensorDataset(test_data, test_labels, transform=transforms.ToTensor())

    elif used_dataset == 1:
        train_dataset = CIFAR100(root='./data', train=True,
                                 download=download_cifar100, transform=transforms.ToTensor())
        test_dataset = CIFAR100(root='./data', train=False,
                                download=download_cifar100, transform=transforms.ToTensor())
    else:
        train_dataset = CIFAR10(root='./data', train=True,
                                download=download_cifar10, transform=transforms.ToTensor())
        test_dataset = CIFAR10(root='./data', train=False,
                               download=download_cifar10, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset),
                                               shuffle=False, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=True, num_workers=2)
    return train_loader, train_dataset, test_loader, test_dataset


def get_nearest_data(train_data: np.ndarray, label_data: np.ndarray, target_img: np.ndarray,
                     target_id: int, num_clusters) -> np.ndarray:
    """
    helper function which uses clustering to find the nearest datapoints for a given target image
    which are then used for the untargeted attack
    :param train_data: numpy array with all training data
    :param target_img: numpy array with the target_image from the testset
    :param num_clusters: the number of clusters in which the data should be clustered

    :return: numpy array with the image ids from the images of the nearest cluster
    """
    print("[ Prepare the data for clustering.. ]")
    start = time.perf_counter()
    # remove the target class data points from the train_data
    train_data = train_data[np.where(label_data != target_id)]

    # append the target image for the clustering and reshape the data
    target_img = torch.unsqueeze(target_img, 0).cpu() # unsqueeze to fit the same dimensions
    data = torch.cat((target_img, train_data), 0).numpy()
    data = np.reshape(data, (data.shape[0], np.prod(data.shape[1:])))

    print("[ Create clustering for the training data.. ]")
    cluster_model = KMeans(n_clusters=num_clusters, verbose=0, n_jobs=4)
    cluster_model.fit(data)

    # obtain the label of our target cluster
    target_cluster_label = cluster_model.labels_[0]

    # search for all ids which are the same label as the target_cluster, excluding the target_image
    nearest_cluster_ids = np.where(cluster_model.labels_[1:] == target_cluster_label)[0]

    end = time.perf_counter()
    duration = np.round(end - start) / 60.
    print(f"[ Created {num_clusters} clusters from {data.shape[0]} datapoints" \
          f" in {duration:0.2f} minutes ]")
    del data
    return nearest_cluster_ids


def gen_pert_dataset(model_name: str, output_name: str, target_class: int,
                     new_class: int, epsilon: float, pgd_iters: int, rand_img: bool,
                     pertube_count: float, loss_fn: int, is_resnet: bool,
                     custom_id: int, device_name: str, used_dataset: int, untargeted: bool,
                     num_clusters: int, layer_cut=1) -> int:
    """
    main function to create a pertubed dataset based on the chosen parameters.
    The algorithms tries to move the new_class (class as which the image should
    get misclassified as) into the direction of the target image in the feature
    space by creating a perturbation which minimizes the difference between the
    activations of the new_class and the target image.

    :param model_name: specifies the name of base model which should get loaded for the attack
    :param output_name: specifies the name as which the dataset should be saved
    :param target_class: the target class for the image which should get missclassified
    :param new_class: the class as which the chosen image should get missclassified
    :param epsilon: specifies the epsilon constraint for the PGD attack
    :param pgd_iters: specifies the iterations for the PGD attack
    :param pertube_count: specifies how much of the original dataset should be manipulated
    :param loss_fn: specifies the loss function (0:BCE, 1:WASSERSTEIN, 2:KLDIV, 3:MinMax)
    :param is_resnet: if flag is true a resnet will be used instead of the normal CNN
    :param custom_id: if set, the algorithm will use the custom_id instead of the
                      most suitable image
    :param device_name: specifies the device on which the dataset will be computed
    :param used_dataset: specifies the dataset used for the attack (0: CIFAR10,
                         1: CIFAR100, 2: TinyImageNet)
    :param layer_cut: specifies how many layers of the classifier should be removed
    :param untargeted: specifies if the attack is untargeted

    :return: if no custom_id is set, the custom_id will be returned. Otherwise the
             ID of the most suitable image will be returned.
    """
    print("[ Initialize.. ]")
    # create a global variable and assign the device which should be used
    global device
    device = device_name
    # initialize several models which are required to calculate the perturbations
    # first is without last dense layers to work on activations
    model = get_model(model_name, device_string=device, layers=layer_cut,
                      is_resnet=is_resnet, used_dataset=used_dataset)
    # second is a normal model used for prediction tasks
    model_complete = get_model(model_name, device_string=device, layers=None,
                               is_resnet=is_resnet, used_dataset=used_dataset)

    # train_loader is used to copy the complete dataset while the raw train and test
    # dataset is used to create pertubed versions of the images
    train_loader, train_dataset, test_loader, test_dataset = get_data(used_dataset=used_dataset)
    # specify the training class size and number of total classes
    # according to the used dataset to normalize the activations
    if used_dataset == 2: # for TinyImageNet
        num_classes = 200
        class_size = 250.
    elif used_dataset == 1: # for CIFAR100
        num_classes = 100
        class_size = 500.
    else: # for CIFAR10
        num_classes = 10
        class_size = 5000.

    # initialize the chosen loss function
    loss_function = None
    if loss_fn == KLDIV:
        loss_function = KLDivLoss()
        loss_class = KLDivLoss()
    if loss_fn == WASSERSTEIN:
        loss_function = WassersteinLoss()
        loss_class = WassersteinLoss()
    if loss_fn == BCE:
        loss_function = F.binary_cross_entropy_with_logits
        loss_class = nn.BCEWithLogitsLoss()
    if loss_fn == MinMax:
        print("[ Compute Complete General Activations for MinMaxLoss ]")
        # create a list with an element for every class and compute the general
        # activations for every class
        class_activations = [None]*num_classes
        for _, (input, target) in tqdm(enumerate(train_dataset)):
            input = input.to(device)
            if class_activations[target] is None:
                class_activations[target] = model(torch.unsqueeze(input, 0)).detach()
            else:
                class_activations[target] += model(torch.unsqueeze(input, 0)).detach()

        class_activations = [class_act.cpu().numpy()/class_size for class_act in class_activations]
        class_activations = torch.Tensor(class_activations).to(device)

        # initialize the Loss class
        loss_function = MinMaxLoss(class_activations, target_class, 0.1)
        loss_class = MinMaxLoss(class_activations, target_class, 0.1)


    target_image_activation = None
    target_image = None
    best_image_id = None

    if custom_id is None:
        # if no custom image id is set
        # iterate over the whole class as which the image should be misclassified
        # sum their activations to obtain their 'general activations'
        print("[ Compute General Activations.. ]")
        general_activation = None
        for _, (input, target) in tqdm(enumerate(train_dataset)):
            input = input.to(device)
            if untargeted:
                if target == target_class:
                    if general_activation is None:
                        general_activation = model(torch.unsqueeze(input, 0)).detach()
                    else:
                        general_activation += model(torch.unsqueeze(input, 0)).detach()
            else:
                if target == new_class:
                    if general_activation is None:
                        general_activation = model(torch.unsqueeze(input, 0)).detach()
                    else:
                        general_activation += model(torch.unsqueeze(input, 0)).detach()
        # normalize the general activations
        general_activation = general_activation / class_size
        general_activation = general_activation.to(device)

        print("[ Compute Target Activations.. ]")
        # if rand_img is True, choose a random image from the target class
        # -> used to evaluate the successrate of an attack for a specific class combination
        # with random target images
        if rand_img or untargeted:
            # choose a random target image
            class_ids = list()
            for idx, (input, target) in enumerate(test_dataset):
                input = input.to(device)
                if target == target_class:
                    # save every id of the right target class
                    class_ids.append(idx)

            # shuffle the class ids to select one randomly
            np.random.shuffle(class_ids)

            # go through the class ids and take the first one for which the original prediction
            # is true (so the class gets recognized correctly on the original model)
            for t_img_id in class_ids:
                input, target = test_dataset[t_img_id]
                input = input.to(device)
                    # create the activations of the target image
                _, prediction = model_complete(torch.unsqueeze(input, 0)).max(1)
                if prediction == target_class:
                    best_image_id = t_img_id
                    target_image_activation = model(torch.unsqueeze(input, 0)).detach()
                    target_image = input
                    break
        else:
            # search for the target image with the smallest difference in its activations
            # compared to the general activations of the new_class. This target image
            # then should be the 'ideal' image to attack, since it's activations already
            # resemble the general activations of the new_class.
            class_input_loss = np.inf
            for idx, (input, target) in enumerate(test_dataset):
                input = input.to(device)
                if target == target_class:
                    # predict the chosen image to check if it gets already misclassified
                    _, prediction = model_complete(torch.unsqueeze(input, 0)).max(1)
                    # generate activations of the current target image and calculate
                    # the difference (loss) to the general activations of the new_class
                    target_activation = model(torch.unsqueeze(input, 0)).detach()
                    current_loss = loss_function(target_activation, general_activation)

                    # checks that the chosen image does not already gets misclassified
                    # while searching for the image with the highest loss
                    # which is therefore the most suitable image to attack
                    if current_loss < class_input_loss and prediction == target_class:
                        best_image_id = idx
                        # create the activations of the target image
                        target_image_activation = model(torch.unsqueeze(input, 0)).detach()
                        target_image = input
                        class_input_loss = current_loss
        print("[ Chose Target with ID: {} ]".format(best_image_id))
    else:
        # otherwise choose the image with the chosen custom image id
        print("[ Compute Target Activations.. ]")
        for idx, (input, target) in enumerate(test_dataset):
            input = input.to(device)
            if idx == custom_id:
                # create the activations of the target image
                target_image_activation = model(torch.unsqueeze(input, 0)).detach()
                best_image_id = idx
                target_image = input
                break
        print("[ Chose Target with ID: {} ]".format(best_image_id))

    # check if the target activations were obtained successfully
    if target_image_activation is None:
        raise RuntimeError("No appropriate target image activations were found. " \
                           " Maybe the model accuracy for this specific class is too low. " \
                           "(The model does not recognize a single image of the class correctly)")

    print("[ Copy Dataset.. ]")
    # create a copy of the dataset so it is possible to just pertube a certain
    # number of images
    new_images, new_labels = list(train_loader)[0]
    new_images_final, _ = list(train_loader)[0]

    # obtain the image ids of the nearest cluster when using the untargeted attack
    nearest_cluster_ids = None
    if untargeted or num_clusters > 0:
        # use the train_dataset copy to generate the cluster embedding and search the nearest
        # cluster for a given target image
        nearest_cluster_ids = get_nearest_data(new_images, new_labels, target_image, target_class,
                                               num_clusters)

    print("[ Building new Dataset.. ]")
    dataset_loss_dict = {}
    current_pertube_count = 0

    # initialize the adversary class
    adversary = L2PGDAttack(model, loss_fn=loss_class, eps=epsilon, nb_iter=pgd_iters,
                            eps_iter=(epsilon/10.), rand_init=True,
                            clip_min=0.0, clip_max=1.0, targeted=True)

    untargeted_adversary = L2PGDAttack(model, loss_fn=loss_class, eps=epsilon, nb_iter=pgd_iters,
                                       eps_iter=(epsilon/10.), rand_init=True,
                                       clip_min=0.0, clip_max=1.0, targeted=False)

    # iterate over the whole dataset and create a perturbed version of every
    # new_class (the class as which the chosen image should be misclassified as)
    # image.
    if untargeted:
        for id in tqdm(nearest_cluster_ids):
            input, target = train_dataset[int(id)]
            input = input.to(device)
            # create a perturbation which uses the activations of the target_class as
            # the target. Therefore, it tries to maximizes the loss between the
            # nearest images and its own class, leading the network to randomly misclassifies it.
            advs = adversary.perturb(torch.unsqueeze(input, 0), target_image_activation).detach()
            new_images[idx] = advs
            activation = model(advs).detach()
            dataset_loss_dict[idx] = loss_function(activation, target_image_activation)

        num_pertubed_imgs = 0
    else:
        for idx, (input, target) in tqdm(enumerate(train_dataset)):
            input = input.to(device)
            if target == new_class:
                # create a perturbation which uses the activations of the new_class as
                # the target. Therefore, it tries to minimize the loss between the
                # two activations by perturbing the image.
                advs = adversary.perturb(torch.unsqueeze(input, 0),
                                         target_image_activation).detach()
                new_images[idx] = advs
                # calculate the loss for every perturbed image and save them in a dict
                if pertube_count != 1.0:
                    activation = model(advs).detach()
                    dataset_loss_dict[idx] = loss_function(activation, target_image_activation)

    # sort the complete dictionary for the lowest loss and the highest if using untargeted mode
    sorted_dataset_loss_dict = sorted(dataset_loss_dict.items(), key=lambda x: x[1],
                                      reverse=untargeted)
    if pertube_count == 1.0 or untargeted:
        # if the whole dataset should get pertubed, just copy the complete dictionary
        # back into the final dataset variable
        new_images_final = new_images
        id_list = []
        for id, _ in sorted_dataset_loss_dict:
            id_list.append(id)
    else:
        # otherwise iterate over the dictionary and choose the N images with the
        # highest loss (therefore the 'best' N images)
        id_list = []
        for id, loss in sorted_dataset_loss_dict:
            if current_pertube_count <= np.floor((pertube_count * len(sorted_dataset_loss_dict))):
                new_images_final[id] = new_images[id]
                id_list.append(id)
                current_pertube_count += 1
            else:
                break

    print("\n[ Saving Dataset: {} ]".format(output_name))
    # check for existing path and save the dataset
    if not os.path.isdir('datasets'):
        os.mkdir('datasets')
    if not os.path.isdir('datasets/{}'.format(output_name)):
        os.mkdir('datasets/{}'.format(output_name))

    torch.save(new_images_final, 'datasets/'+str(output_name)+'/ims_'+str(output_name))
    torch.save(new_labels, 'datasets/'+str(output_name)+'/lab_'+str(output_name))
    # additionally save the image IDs, so it is possible to distinguish the perturbed
    # and original images to create plots for comparisons
    torch.save(id_list, 'datasets/'+str(output_name)+'/ids_'+str(output_name))

    return best_image_id
