"""main file to run training, attack and evaluation"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time
import os
import socket
import datetime
import argparse
import torch
import numpy as np
from custom_modules.dictionaries import get_dicts
from train_methods.training import train
from evaluation_methods.layer import analyze_layers
from evaluation_methods.single_model import evaluate_single_model
from evaluation_methods.single_class import evaluate_single_class
from evaluation_methods.check_utils import check_success, save_dataframe, get_best_parameters
from evaluation_methods.check_utils import plot_attack_results
from dataset_generation_methods.single_image import gen_pert_dataset

torch.backends.cudnn.benchmark = True

# ---------------- Static Parameters & Dictionaries -----------------------
BCE, WASSERSTEIN, KLDIV, MinMax = 0, 1, 2, 3
PGD_ITERS = 100
EPOCHS = 100
LR = 0.1
BATCH_SIZE = 128


def main(eps: int, gpu: int, pert: int, loss_fn: int, layer_cuts: int, resnet: bool,
         target_class: int, new_class: int, dataset: int, image_id: int, eva: bool,
         transfer: bool, rand: bool, iters: int, best: bool, untargeted: bool,
         cluster: int) -> None:
    """main method to start training and evaluation
    procedures by iteration through all training parameters
    """
    start = time.perf_counter()
    # set device properly
    if gpu == 0:
        DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if gpu == 1:
        DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # if no parameters were provided, they have to be wrapped into a list to be iterable
    if isinstance(eps, (int, float)):
        eps = list([eps])
    if isinstance(pert, (int, float)):
        pert = list([pert])
    if isinstance(loss_fn, (int, float)):
        loss_fn = list([loss_fn])
    if isinstance(layer_cuts, (int, float)):
        layer_cuts = list([layer_cuts])
    if target_class is None:
        target_class = list([target_class])
    if new_class is None:
        new_class = list([new_class])
    if cluster is None:
        cluster = 0

    # set dictionaries and model names
    class_dict, class_dict_rev, loss_dict = get_dicts(used_dataset=dataset)

    # specifies the basic model which is used to create the adversaries
    if dataset == 2:
        cluster_size = 49751
        if resnet:
            base_model_name = 'resnet_tinyimagenet'
        else:
            base_model_name = 'basic_tinyimagenet'
    elif dataset == 1:
        cluster_size = 49501
        if resnet:
            base_model_name = 'resnet_cifar100'
        else:
            base_model_name = 'basic_cifar100'
    else:
        cluster_size = 45001
        if resnet:
            base_model_name = 'resnet_cifar10'
        else:
            base_model_name = 'basic_cifar10'

    if best:
        # if best is set, load and use the successful parameters for this class for testing
        eps, loss_fn, pert, layer_cuts = get_best_parameters(target_class[0], new_class[0],
                                                             base_model_name)

    # calculate the total attacks for one constellation so it is possible to
    # calculate the success ratio later on
    num_total_attacks = len(layer_cuts)*len(pert)*len(loss_fn)*len(eps)
    if iters:
        if target_class[0] is not None:
            # normal attacks times the iterations for same class tests when classes are not random
            num_total_attacks *= iters
        # also duplicate the class combinations for more iterations
        target_class = target_class * iters
        new_class = new_class *iters

    # set flag if a custom image should be used
    if image_id:
        assert len(image_id) == len(target_class), "Not enough image_ids provided!"
    else:
        image_id = [None]*len(target_class)

    # check if a normal cifar10/100 model already exists
    # if not, train a new one
    if not os.path.isdir('./model_saves/{}'.format(base_model_name)):
        print("[ No base model found. Create a new one: {} ]".format(base_model_name))
        train(epochs=EPOCHS,
              learning_rate=LR,
              output_name=base_model_name,
              data_suffix=None,
              batch_size=BATCH_SIZE,
              device_name=DEVICE,
              is_resnet=resnet,
              used_dataset=dataset,
              use_transfer=transfer,
              data_augmentation=True)

    # print a summary of the chosen arguments
    print("\n\n\n"+"#"*50)
    print("# " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print("# System: {} CPU cores with {} GPUs on {}".format(torch.get_num_threads(),
                                                             torch.cuda.device_count(),
                                                             socket.gethostname()
                                                             ))
    if DEVICE == 'cpu':
        print("# Using: CPU with ID {}".format(DEVICE))
    else:
        print("# Using: {} with ID {}".format(torch.cuda.get_device_name(device=DEVICE), DEVICE))
    print("# Eps: ", str(eps))
    print("# Perurbation_Count: {}".format(pert))
    print("# Loss_Function: {}".format(loss_dict[loss_fn[0]]))
    print("# Layer_Cuts: {}".format(layer_cuts))
    if target_class[0] is not None:
        print("# Target_Classes: {} ({})".format(target_class,
                                                 [class_dict_rev[i] for i in target_class]))
    else:
        print("# Target_Classes: Random (?)")
    if new_class[0] is not None:
        print("# Poison_Classes: {} ({})".format(new_class,
                                                 [class_dict_rev[i] for i in new_class]))
    elif untargeted:
        print("# Poison_Class: Untargeted")
    else:
        print("# Poison_Classes: Random (?)")
    print("# Clusters: {} with size: {}".format(cluster, cluster_size//cluster))
    print("# Image_Id: {}".format("Random" if rand else image_id))
    print("# Using Resnet: {}".format(resnet))
    print("# Dataset: {}".format('TinyImageNet' if dataset == 2 else
                                 ('CIFAR100' if dataset == 1 else 'CIFAR10')))
    print("# Transfer-Learning: {}".format(transfer))
    print("# Total Attacks: {}".format(num_total_attacks*len(target_class)))
    print("#"*50)

    # counter for the current attack iteration
    current_attack_iter = 0
    # counter for successful attacks of a class combination
    successful_attacks = 0
# ------------- Iterate over the given set of parameters ----------------------
    for t_class, n_class, i_image_id in zip(target_class, new_class, image_id):
        current_image_id = i_image_id
        # if no target and new class is chosen, select them randomly
        if t_class is None and n_class is None:
            t_class = class_dict[np.random.randint(len(class_dict))]
            print(">> selected target class: {} randomly".format(t_class))
            while True:
                n_class = class_dict[np.random.randint(len(class_dict))]
                if n_class is not t_class:
                    break
            if not untargeted:
                print(">> selected poison class: {} randomly".format(n_class))

        # set target and new class from class dict
        t_class = class_dict_rev[t_class]
        n_class = class_dict_rev[n_class]

        data_suffix_string = ""
        t_class_string = class_dict[t_class].lower()
        if untargeted:
            n_class_string = "untargeted"
        else:
            n_class_string = class_dict[n_class].lower()

        if transfer:
            data_suffix_string += "_transfer"
        if rand:
            data_suffix_string += "_rand"
        # create path strings and create directories to save the plot results
        save_path = "{}_{}_to_{}{}".format(base_model_name, t_class_string, n_class_string,
                                           data_suffix_string)
        result_path = 'results/{}_results'.format(save_path)
        if not os.path.isdir(result_path):
            if not os.path.isdir('results/'):
                os.mkdir('results/')
            os.mkdir(result_path)
        print("Name: {}".format(save_path))

        if not best:
            # reset the counters for every class constellation if you dont use 'best' option
            successful_attacks = 0
            current_attack_iter = 0

    # ---------------- iterate over the parameters for each constellation of classes ---------------
        for i_layer in layer_cuts:
            for i_pert_count in pert:
                for i_loss_fn in loss_fn:
                    for i_eps in eps:
                        current_attack_iter += 1
                        print("\n[ Attack: {}/{} ]".format(current_attack_iter, num_total_attacks))
                        print("[ pert_count: {} | loss_fn: {} | eps: {} | {} layer ]\n".format(
                            i_pert_count, loss_dict[i_loss_fn], i_eps, i_layer))

                        dataset_name = "{}_{}_to_{}_{}_{}_pertcount_{}_eps_{}layer{}".format(
                            'resnet' if resnet else 'basic', t_class_string, n_class_string,
                            loss_dict[i_loss_fn], i_pert_count, i_eps, i_layer, data_suffix_string)

    # ----------------- skip training if the evaluation flag is set ---------------
                        if not eva:
                            current_image_id = gen_pert_dataset(model_name=base_model_name,
                                                                output_name=dataset_name,
                                                                target_class=t_class,
                                                                new_class=n_class,
                                                                epsilon=i_eps,
                                                                rand_img=rand,
                                                                pgd_iters=PGD_ITERS,
                                                                pertube_count=i_pert_count,
                                                                loss_fn=i_loss_fn,
                                                                custom_id=current_image_id,
                                                                device_name=DEVICE,
                                                                used_dataset=dataset,
                                                                is_resnet=resnet,
                                                                layer_cut=i_layer,
                                                                untargeted=untargeted,
                                                                num_clusters=cluster)

                            train(epochs=EPOCHS,
                                  learning_rate=LR,
                                  output_name="{}_{}".format(base_model_name, dataset_name),
                                  data_suffix=dataset_name,
                                  batch_size=BATCH_SIZE,
                                  device_name=DEVICE,
                                  is_resnet=resnet,
                                  used_dataset=dataset,
                                  use_transfer=transfer,
                                  data_augmentation=True)
                        else:
                            assert current_image_id is not None, "image_id is not set!"

    # --------------------------------- analyze the model -------------------------
                        # check if the attack was successful
                        success_flag = check_success(new_class=n_class,
                                                     target_id=current_image_id,
                                                     is_resnet=resnet,
                                                     model_name="{}_{}".format(\
                                                     base_model_name, dataset_name),
                                                     used_dataset=dataset,
                                                     untargeted=untargeted)

                        # analyze the whole layer activation of the penultimate and last layer
                        analyze_layers(epsilon=i_eps,
                                       pgd_iters=PGD_ITERS,
                                       target_class=t_class,
                                       new_class=n_class,
                                       save_path=result_path,
                                       model_name="{}_{}".format(base_model_name, dataset_name),
                                       pert_count=i_pert_count,
                                       loss_fn=i_loss_fn,
                                       device_name=DEVICE,
                                       layer_cut=i_layer,
                                       is_resnet=resnet,
                                       used_dataset=dataset,
                                       target_id=current_image_id)

                        # evaluate the performance of the model for target and new class and
                        # visualize the results as a plot.
                        acc_single = evaluate_single_class(model_name="{}_{}".format(\
                                                           base_model_name, dataset_name),
                                                           save_path=result_path,
                                                           target_class=t_class,
                                                           new_class=n_class,
                                                           epsilon=i_eps,
                                                           pgd_iters=PGD_ITERS,
                                                           pert_count=i_pert_count,
                                                           loss_function=i_loss_fn,
                                                           device_name=DEVICE,
                                                           is_resnet=resnet,
                                                           used_dataset=dataset,
                                                           layer_cut=i_layer)

                        # Evaluate the normal and adversarial model accuracy using unmodified
                        # and pertubed images of the whole cifar10 test dataset.
                        acc_whole = evaluate_single_model(model_name="{}_{}".format(\
                                                          base_model_name, dataset_name),
                                                          save_path=result_path,
                                                          target_class=t_class,
                                                          new_class=n_class,
                                                          epsilon=i_eps,
                                                          pgd_iters=PGD_ITERS,
                                                          pert_count=i_pert_count,
                                                          loss_function=i_loss_fn,
                                                          device_name=DEVICE,
                                                          is_resnet=resnet,
                                                          used_dataset=dataset,
                                                          layer_cut=i_layer)

                        # save accuracies and successful parameters as a dataframe
                        if success_flag:
                            successful_attacks += 1

                        save_dataframe(epsilon=i_eps,
                                       layer_cuts=i_layer,
                                       target_class=t_class_string,
                                       new_class=n_class_string,
                                       loss_fn=i_loss_fn,
                                       pert_count=i_pert_count,
                                       current_total_attacks=num_total_attacks,
                                       successful_attacks=successful_attacks,
                                       acc_single=acc_single,
                                       acc_whole=acc_whole,
                                       rand_img=rand,
                                       best_img=best,
                                       prefix=base_model_name,
                                       num_clusters=cluster)

    # plot the resulting accuracies for all existing successful attack parameters
    plot_attack_results(base_model_name)
    print("[ {}/{} Attacks in total were successful ]".format(successful_attacks,
                                                              num_total_attacks*len(target_class)))
    print("finished: [ {} ]".format(dataset_name))
    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"Computation time: {duration:0.4f} hours")


# ---------------------------------------main hook-------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", "-g", help="GPU", type=int, default=0)
    parser.add_argument("--eps", "-e", help="Epsilon", nargs='+', type=float, default=0.5)
    parser.add_argument("--pert", "-p", help="Pert. Percentage", nargs='+', type=float, default=0.5)
    parser.add_argument("--loss_fn", "-l", help="Loss Function", type=int, nargs='+', default=2)
    parser.add_argument("--layer_cuts", "-c", help="i_layer Cuts", type=int, nargs='+', default=1)
    parser.add_argument("--target_class", "-t", help="Target Class", type=str,
                        nargs='+', required=False)
    parser.add_argument("--new_class", "-n", help="New Class", type=str, nargs='+', required=False)
    parser.add_argument("--dataset", "-d", help="specifies the used origin dataset",
                        type=int, default=0, required=False)
    parser.add_argument("--resnet", "-r", help="uses resnet instead of the normal cnn",
                        action='store_true', default=False, required=False)
    parser.add_argument("--transfer", "-f", help="use transfer learning to train only the fc layer",
                        action='store_true', default=False, required=False)
    parser.add_argument("--image_id", "-i", help="Custom Best Image ID", type=int,
                        nargs='+', default=None)
    parser.add_argument("--eva", "-v", help="skip train, just evaluate", action='store_true',
                        required=False)
    parser.add_argument("--rand", "-a", help="use random images as target img", action='store_true',
                        default=False)
    parser.add_argument("--iters", "-s", help="iters for same class tests", type=int, default=None)
    parser.add_argument("--best", "-b", help="uses the best parameters for a class const",
                        action='store_true', required=False)
    parser.add_argument("--untargeted", "-u", help="performs an untargeted attack",
                        action='store_true', required=False)
    parser.add_argument("--cluster", "-cl", help="specifies how many clusters should be used",
                        type=int, default=20, required=False)

    args = parser.parse_args()
    main(**vars(args))
