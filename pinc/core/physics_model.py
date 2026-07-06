from abc import ABC, abstractmethod
import torch


class PhysicsModel(ABC):
    """
    Abstract interface for all dynamical systems.
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def control_dim(self) -> int:
        pass

    @abstractmethod
    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Returns dx/dt
        """
        pass

    def residual(self, x, dx, u):
        """
        Physics residual: dx/dt - f(x,u)
        """
        return dx - self.dynamics(x, u)