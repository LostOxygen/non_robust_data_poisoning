"""library module for the used CNN model"""
import torch
import torch.nn as nn

def _weights_init(m) -> None:
    """
    helper function to apply he-initialization on the network weights
    :param m: model layer on which the initialization should be applied

    :return: None
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class CNN(nn.Module):
    """a standard CNN model which performs well on the CIFAR10 dataset"""
    def __init__(self, num_classes: int = 10, img_size: int = 32) -> None:
        super(CNN, self).__init__()

        if img_size == 32:
            classifier_input = 4096
        else:
            classifier_input = 16384

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(classifier_input, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward function of the model"""
        x = self.conv_layer(x)
        x = torch.flatten(x, 1) #Flatten
        x = self.fc(x)
        return x
