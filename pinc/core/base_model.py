from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Base interface for PINN, PINC, DeepONet, etc.
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        return self.forward(*args, **kwargs)