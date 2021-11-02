"""library module for custom loss functions"""
import torch
import torch.nn.functional as F


class KLDivLoss(torch.nn.Module):
    """standard kullback leibler divergence loss as described in:
       https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    """
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, y: float, y_hat: float) -> float:
        """forward method to calculate the loss for a given prediction and label"""
        return F.kl_div(F.log_softmax(y, dim=1),
                        F.softmax(y_hat, dim=1),
                        None, None, reduction='sum')


class MinMaxLoss(torch.nn.Module):
    """a loss that tries to minimize the loss to a chosen class
       while maximizing the loss to every other class
    """
    def __init__(self, class_activations: float, target_class: int, weight: float):
        super(MinMaxLoss, self).__init__()
        self.target_class = target_class
        self.class_activations = class_activations
        self.loss_weight = weight

    def forward(self, y: float, y_hat: float) -> float:
        """forward method to calculate the loss for a given prediction and label
           y is the image which gets manipulated, while y_hat is the target"""
        loss = F.kl_div(F.log_softmax(y, dim=1), F.softmax(y_hat, dim=1),
                        None, None, reduction='sum')

        for class_id in range(0, 10):
            #if class_id is not self.target_class:
            if class_id == 3: #Cat class
                ce_diff = F.kl_div(F.log_softmax(y, dim=1),
                                   F.softmax(self.class_activations[class_id], dim=1),
                                   None, None, reduction='sum')
                loss += ce_diff * self.loss_weight
        return loss

class WassersteinLoss(torch.nn.Module):
    """loss function based on the wasserstein distance described by:
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
    """
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, tensor_a: float, tensor_b: float) -> float:
        """forward method to calculate the loss for a given prediction and label"""
        tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
        tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)

        cdf_tensor_a = torch.cumsum(tensor_a, dim=-1)
        cdf_tensor_b = torch.cumsum(tensor_b, dim=-1)

        cdf_distance = torch.sum(torch.abs(torch.sub(cdf_tensor_a, cdf_tensor_b)), dim=-1)
        cdf_loss = cdf_distance.mean() * 0.1

        return cdf_loss
