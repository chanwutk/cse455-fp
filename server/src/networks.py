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


class SavableAlexNet(PreMadeNetwork):
    def __init__(self, num_classes: int):
        super(SavableAlexNet, self).__init__(
            models.alexnet(pretrained=False, num_classes=num_classes)
        )


class SavableVGG(PreMadeNetwork):
    def __init__(self, num_classes: int):
        super(SavableVGG, self).__init__(
            models.vgg19_bn(pretrained=False, num_classes=num_classes)
        )


class TraceableSequential(nn.Sequential):
    def __init__(self, args, traceable=False):
        super(TraceableSequential, self).__init__(*args)
        self.traceable = traceable
        self.traces = []

    def forward(self, input_tensor: torch.Tensor):
        if self.traceable:
            self.traces = []
        for module in self._modules.values():
            module_input = None
            if self.traceable:
                module_input = input_tensor.detach()
            input_tensor = module(input_tensor)
            if self.traceable:
                module_output = input_tensor.detach()
                weight = None
                if hasattr(module, "weight") and len(module.weight.size()) == 4:
                    weight = module.weight.detach()
                self.traces.append((module_input, module_output, module, weight))
        return input_tensor

    def get_traces(self):
        return self.traces

    def set_traceable(self, traceable: bool = True):
        self.traceable = traceable

    def __str__(self):
        return super(TraceableSequential, self).__str__()


class GenericTraceableNet(BaseSavableNet):
    def __init__(self, net: nn.Module, seq_attr: str, traceable: bool = False):
        super(GenericTraceableNet, self).__init__()
        if getattr(net, seq_attr).__class__.__name__ != "Sequential":
            raise Exception("the attribute seq_attr of net should be Sequential")

        traceable_seq = TraceableSequential(
            list(getattr(net, seq_attr).modules())[1:], traceable=traceable
        )
        setattr(net, seq_attr, traceable_seq)
        self.net = net
        self.traceable = traceable
        self.seq_attr = seq_attr

    def forward(self, input_tensor: torch.Tensor):
        return self.net(input_tensor)

    def set_traceable(self, traceable: bool = True):
        self.traceable = traceable
        getattr(self.net, self.seq_attr).set_traceable(traceable)

    def get_traces(self):
        return getattr(self.net, self.seq_attr).get_traces()
