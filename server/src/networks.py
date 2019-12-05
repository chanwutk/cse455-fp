import torch
import torch.nn as nn
import torch.nn.functional as F
import src.pt_util as pt_util
import torchvision.models as models


class BaseSavableNet(nn.Module):
    def __init__(self):
        super(BaseSavableNet, self).__init__()
        self.__best_accuracy_saved = None

    def classify(self, input_tensor: torch.Tensor):
        return F.softmax(self.forward(input_tensor), dim=1)

    def loss(
        self, prediction: torch.Tensor, label: torch.Tensor, reduction: str = "mean"
    ):
        loss_val = F.cross_entropy(prediction, label.squeeze(), reduction=reduction)
        return loss_val

    def save_model(self, file_path: str, num_to_keep: int = 1):
        pt_util.save(self, file_path, num_to_keep)

    def save_best_model(self, accuracy: float, file_path: str, num_to_keep: int = 1):
        if self.__best_accuracy_saved is None or self.__best_accuracy_saved < accuracy:
            self.__best_accuracy_saved = accuracy
            self.save_model(file_path, num_to_keep)

    def load_model(self, file_path: str):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path: str):
        return pt_util.restore_latest(self, dir_path)


class PreMadeNetwork(BaseSavableNet):
    def __init__(self, net: nn.Module):
        super(PreMadeNetwork, self).__init__()
        self.net = net

    def forward(self, input_tensor: torch.Tensor):
        return self.net.forward(input_tensor)


class PneumoniaAlexNet(PreMadeNetwork):
    def __init__(self):
        super(PneumoniaAlexNet, self).__init__(
            models.alexnet(pretrained=False, num_classes=2)
        )


class PneumoniaVGG(PreMadeNetwork):
    def __init__(self):
        super(PneumoniaVGG, self).__init__(
            models.vgg19_bn(pretrained=False, num_classes=2)
        )
