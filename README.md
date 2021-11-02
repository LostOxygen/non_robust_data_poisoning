# Data Poisoning based on Adversarial Attacks using Non-Robust Features
### Usage
```python
python main.py [-h] [--gpu | -g GPU]  [--eps |-e EPSILON] [--pert | -p PERTURBATION_PERCENTAGE] [--loss_fn | -l LOSS_FUNCTION] [--layer_cuts | -c LAYER_CUTS] [--target_class | -t TARGET_CLASS] [--new_class | -n NEW_CLASS] [-v | --eva] [--dataset | -d DATASET] [--resnet | -m] [--transfer | -f] [--rand | -a] [--iters | -s ITERATIONS]
```
### Arguments
| Argument | Type | Description|
|----------|------|------------|
| -h, --help | None| shows argument help message |
| -g, --gpu | INT | specifies which GPU should be used [0, 1] |
| -e, --eps | INT | specifies the epsilon value which is used to perturb the images |
| -p, --pert | FLOAT | specifies how much of the dataset (in %) gets perturbed |
| -l, --loss_fn | INT | specifies the loss function: [0] BCE, [1] Wasserstein, [2] KL-Div, [3] MinMax |
| -c, --layer_cuts | INT | specifies the dense layer(s) (counting from last to first) from which the activations are obtained |
| -t, --target_class | INT | specifies the target class (from which the 'best' image will be used for misclassification) |
| -n, --new_class | INT | specifies the class as which the chosen image gets misclassified |
| -i, --image_id | INT | specifies the ID of a certain image which will be misclassified instead of the 'best' target class image |
| -v, --eval | BOOL | skips the training phase and only runs the evaluation. Needs --image_id to be set |
| -d, --dataset | INT | specifies the used dataset: [0] Cifar10, [1] Cifar100, [2] TinyImageNet |
| -m, --is_resnet | BOOL | set flag if the resnet model should be used |
| -f, --transfer | BOOL | set flag if transfer learning should be used (Freeze the feature extraction and only train the classifier on the new dataset) |
| -a, --rand | BOOL | set flag if a random target image instead of the most suitable one should be used |
| -s, --iters | INT | duplicates the given target and new class to test more iterations of complete attacks on them. Makes passing a list of same classes obsolete |
| -b, --best | BOOL | set flag if the successful attack parameters for a given class combination should be loaded |
| -u, --untargeted | BOOL | set flag to perform an untargeted attack on the target class |
| -cl, --cluster | INT | specifies the number of clusters in which the training data is divided for the untargeted attack |

### Examples

```python
python main.py --gpu 0 --eps 2 1 0.75 0.5 0.25 0.1 --pert 0.5 --loss_fn 2 --layer_cuts 1 2 --dataset 0 --target_class "deer" --new_class "horse"
```
Would use **deer** as the target class and **horse** as the new class to create 12 datasets. Six datasets with ​epsilon = [2, 1, 0.75, 0.5, 0.25, 0.1] and the activations from the last dense layer and six datasets with the same epsilon values but the activations from the penultimate dense layer. Both datasets contain 50% perturbed images and the generation as well as the training is performed on GPU:0. The model used is the standard CNN while the dataset is a unmodified CIFAR10 dataset.

```python
python main.py --gpu 1 --dataset 1 --target_class "bee" --new_class "beetle" --resnet --transfer --rand --iters 10 --best
```
Would load the attack parameters from ```results/attack_results.pkl``` for the chosen class combination and would choose **10 times** a **random** target image to test these parameters on.


### Untargeted Attack Test-Calls
```python
python3 main.py --gpu 0 --dataset 0 --eps 0.5 --pert 1.0 --loss_fn 2 --resnet --transfer --untargeted --rand --cluster 1 --iters 10
```

### Download TinyImageNet
```bash
wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
```
