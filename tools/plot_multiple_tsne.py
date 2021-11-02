import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import os
import numpy as np
from tqdm import tqdm
from models import *
from custom_modules import TensorDataset
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_dict = {0:"Airplane", 1:"Auto", 2:"Bird", 3:"Cat", 4:"Deer", 5:"Dog", 6:"Frog", 7:"Horse", 8:"Ship", 9:"Truck"}
AIRPLANE, AUTO, BIRD, CAT, DEER, DOG, FROG, HORSE, SHIP, TRUCK = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

def get_model(path0, path1):
    basic_net = CNN()
    checkpoint = torch.load('./model_saves/basic_training_'+str(path0)+'/basic_training_'+str(path1))
    #basic_net.fc_layer = nn.Sequential(nn.Identity())
    basic_net.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True)
    )
    basic_net.load_state_dict(checkpoint['net'], strict=False)
    basic_net = basic_net.to(device)
    basic_net.eval()
    return basic_net

# ---------------------------------------------------
def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


if __name__ == "__main__":
    target_class = DEER
    new_class = HORSE
    normal_dataset = False
    data_suffix = "single_deer_to_horse"
    target_image_id = 9035
    poison_ids = None


    print("[ Initialize.. ]")
    model0 = get_model(data_suffix, data_suffix+"_1")
    model1 = get_model(data_suffix, data_suffix+"_25")
    model2 = get_model(data_suffix, data_suffix+"_50")
    model3 = get_model(data_suffix, data_suffix+"_75")
    model4 = get_model(data_suffix, data_suffix+"_100")

    data_path = "datasets/"+str(data_suffix)+"/"

    if normal_dataset:
        poison_ids = torch.load(os.path.join(data_path, f"CIFAR_ids_"+str(data_suffix)))
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=1)
        imgs, labels = list(train_loader)[0]
        train_data = torch.cat((torch.tensor(imgs), torch.tensor(test_dataset[target_image_id][0]).unsqueeze(0)))
        train_labels = torch.cat((torch.tensor(labels), torch.tensor(test_dataset[target_image_id][1]).unsqueeze(0)))
    else:
        poison_ids = torch.load(os.path.join(data_path, f"CIFAR_ids_"+str(data_suffix)))
        train_data = torch.load(os.path.join(data_path, f"CIFAR_ims_"+str(data_suffix)))
        train_labels = torch.load(os.path.join(data_path, f"CIFAR_lab_"+str(data_suffix)))
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
        train_data = torch.cat((train_data, torch.tensor(test_dataset[target_image_id][0]).unsqueeze(0)))
        train_labels = torch.cat((train_labels, torch.tensor(test_dataset[target_image_id][1]).unsqueeze(0)))


    dataset = TensorDataset(train_data, train_labels, transform=transforms.ToTensor())
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    whole_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=1)


    # features0, features1, features2, features3, features4 = None, None, None, None, None
    # for idx, (input, target) in tqdm(enumerate(dataset_loader), desc="Running Model Inference"):
    #     input = input.to(device)
    #     with torch.no_grad():
    #         output0 = model0.forward(input)
    #         output1 = model1.forward(input)
    #         output2 = model2.forward(input)
    #         output3 = model3.forward(input)
    #         output4 = model4.forward(input)
    #
    #     current_features0 = output0.cpu().numpy()
    #     current_features1 = output1.cpu().numpy()
    #     current_features2 = output2.cpu().numpy()
    #     current_features3 = output3.cpu().numpy()
    #     current_features4 = output4.cpu().numpy()
    #
    #     if features0 is not None:
    #         features0 = np.concatenate((features0, current_features0))
    #         features1 = np.concatenate((features1, current_features1))
    #         features2 = np.concatenate((features2, current_features2))
    #         features3 = np.concatenate((features3, current_features3))
    #         features4 = np.concatenate((features4, current_features4))
    #     else:
    #         features0 = current_features0
    #         features1 = current_features1
    #         features2 = current_features2
    #         features3 = current_features3
    #         features4 = current_features4
    #
    #
    # print("[ Running TSNE ]")
    # tsne0 = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features0)
    # tsne1 = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features1)
    # tsne2 = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features2)
    # tsne3 = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features3)
    # tsne4 = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features4)

    tsne0 = torch.load('results/t_SNE/tsne0_'+str(data_suffix))
    tsne1 = torch.load('results/t_SNE/tsne1_'+str(data_suffix))
    tsne2 = torch.load('results/t_SNE/tsne2_'+str(data_suffix))
    tsne3 = torch.load('results/t_SNE/tsne3_'+str(data_suffix))
    tsne4 = torch.load('results/t_SNE/tsne4_'+str(data_suffix))

    # try:
    #     torch.save(tsne0, 'results/t_SNE/tsne0_'+str(data_suffix))
    #     torch.save(tsne1, 'results/t_SNE/tsne1_'+str(data_suffix))
    #     torch.save(tsne2, 'results/t_SNE/tsne2_'+str(data_suffix))
    #     torch.save(tsne3, 'results/t_SNE/tsne3_'+str(data_suffix))
    #     torch.save(tsne4, 'results/t_SNE/tsne4_'+str(data_suffix))
    # except Exception as e:
    #     print("Couldn't save tsne data: " + str(e))

    tx0, ty0, tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4 = tsne0[:, 0], tsne0[:, 1], tsne1[:, 0], tsne1[:, 1], tsne2[:, 0], tsne2[:, 1], tsne3[:, 0], tsne3[:, 1], tsne4[:, 0], tsne4[:, 1]

    tx0, ty0 = scale_to_01_range(tx0), scale_to_01_range(ty0)
    tx1, ty1 = scale_to_01_range(tx1), scale_to_01_range(ty1)
    tx2, ty2 = scale_to_01_range(tx2), scale_to_01_range(ty2)
    tx3, ty3 = scale_to_01_range(tx3), scale_to_01_range(ty3)
    tx4, ty4 = scale_to_01_range(tx4), scale_to_01_range(ty4)

    fig = plt.figure()
    fig.suptitle("t_SNE | "+str(data_suffix)+" | $\epsilon=0.5$ | iters=100 | 50% Perturbation | KLDiv | without Softmax | without last layer | CIFAR10 ")

    ax0 = fig.add_subplot(511)
    ax1 = fig.add_subplot(512)
    ax2 = fig.add_subplot(513)
    ax3 = fig.add_subplot(514)
    ax4 = fig.add_subplot(515)

    print("[ Visualize.. ]")
    classes = [0,1,2,3,4,5,6,7,8,9]
    _, labels = list(whole_loader)[0]

    for single_class in classes:
        indices = [i for i, l in enumerate(labels) if l == single_class] #erstellt indizes der jeweiligen klasse zum plotten

        if single_class == target_class:
            current_tx0, current_ty0 = np.take(tx0, indices), np.take(ty0, indices)
            current_tx1, current_ty1 = np.take(tx1, indices), np.take(ty1, indices)
            current_tx2, current_ty2 = np.take(tx2, indices), np.take(ty2, indices)
            current_tx3, current_ty3 = np.take(tx3, indices), np.take(ty3, indices)
            current_tx4, current_ty4 = np.take(tx4, indices), np.take(ty4, indices)

            ax0.scatter(current_tx0, current_ty0, c='lightsteelblue', label=class_dict[single_class], edgecolors='none', alpha=0.5)
            ax1.scatter(current_tx1, current_ty1, c='lightsteelblue', label=class_dict[single_class], edgecolors='none', alpha=0.5)
            ax2.scatter(current_tx2, current_ty2, c='lightsteelblue', label=class_dict[single_class], edgecolors='none', alpha=0.5)
            ax3.scatter(current_tx3, current_ty3, c='lightsteelblue', label=class_dict[single_class], edgecolors='none', alpha=0.5)
            ax4.scatter(current_tx4, current_ty4, c='lightsteelblue', label=class_dict[single_class], edgecolors='none', alpha=0.5)

        if single_class == new_class:
            current_tx0, current_ty0 = np.take(tx0, indices), np.take(ty0, indices)
            current_tx1, current_ty1 = np.take(tx1, indices), np.take(ty1, indices)
            current_tx2, current_ty2 = np.take(tx2, indices), np.take(ty2, indices)
            current_tx3, current_ty3 = np.take(tx3, indices), np.take(ty3, indices)
            current_tx4, current_ty4 = np.take(tx4, indices), np.take(ty4, indices)

            ax0.scatter(current_tx0, current_ty0, c='mediumseagreen', label=class_dict[single_class], edgecolors='none', alpha=0.5)
            ax1.scatter(current_tx1, current_ty1, c='mediumseagreen', label=class_dict[single_class], edgecolors='none', alpha=0.5)
            ax2.scatter(current_tx2, current_ty2, c='mediumseagreen', label=class_dict[single_class], edgecolors='none', alpha=0.5)
            ax3.scatter(current_tx3, current_ty3, c='mediumseagreen', label=class_dict[single_class], edgecolors='none', alpha=0.5)
            ax4.scatter(current_tx4, current_ty4, c='mediumseagreen', label=class_dict[single_class], edgecolors='none', alpha=0.5)


    poisons_tx0, poisons_ty0 = np.take(tx0, poison_ids), np.take(ty0, poison_ids)
    poisons_tx1, poisons_ty1 = np.take(tx1, poison_ids), np.take(ty1, poison_ids)
    poisons_tx2, poisons_ty2 = np.take(tx2, poison_ids), np.take(ty2, poison_ids)
    poisons_tx3, poisons_ty3 = np.take(tx3, poison_ids), np.take(ty3, poison_ids)
    poisons_tx4, poisons_ty4 = np.take(tx4, poison_ids), np.take(ty4, poison_ids)

    ax0.scatter(poisons_tx0, poisons_ty0, c='red', label='poison', alpha=0.4, s=20)
    ax1.scatter(poisons_tx1, poisons_ty1, c='red', label='poison', alpha=0.4, s=20)
    ax2.scatter(poisons_tx2, poisons_ty2, c='red', label='poison', alpha=0.4, s=20)
    ax3.scatter(poisons_tx3, poisons_ty3, c='red', label='poison', alpha=0.4, s=20)
    ax4.scatter(poisons_tx4, poisons_ty4, c='red', label='poison', alpha=0.4, s=20)

    target_tx0, target_ty0 = np.take(tx0, -1), np.take(ty0, -1)
    target_tx1, target_ty1 = np.take(tx1, -1), np.take(ty1, -1)
    target_tx2, target_ty2 = np.take(tx2, -1), np.take(ty2, -1)
    target_tx3, target_ty3 = np.take(tx3, -1), np.take(ty3, -1)
    target_tx4, target_ty4 = np.take(tx4, -1), np.take(ty4, -1)

    ax0.scatter(target_tx0, target_ty0, c='blue', label='target', marker="^", s=300)
    ax1.scatter(target_tx1, target_ty1, c='blue', label='target', marker="^", s=300)
    ax2.scatter(target_tx2, target_ty2, c='blue', label='target', marker="^", s=300)
    ax3.scatter(target_tx3, target_ty3, c='blue', label='target', marker="^", s=300)
    ax4.scatter(target_tx4, target_ty4, c='blue', label='target', marker="^", s=300)

    ax0.set_title("Epoch 0")
    ax1.set_title("Epoch 25")
    ax2.set_title("Epoch 50")
    ax3.set_title("Epoch 75")
    ax4.set_title("Epoch 100")

    ax0.legend(loc='right')
    ax1.legend(loc='right')
    ax2.legend(loc='right')
    ax3.legend(loc='right')
    ax4.legend(loc='right')

    fig.set_size_inches(15, 10)
    #plt.savefig('results/'+ str(data_suffix) +'_results/tsne_'+ str(data_suffix) +'.png')
    #plt.rcParams.update({'font.size': 16})
    plt.show()
