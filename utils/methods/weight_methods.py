import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from utils.methods.min_norm_solvers import MinNormSolver, gradient_normalizers

EPS = 1e-8 # for numerical stability


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device, max_norm = 1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class FAMO(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 1e-5,
        w_lr: float = 0.025,
        task_weights: Union[List[float], torch.Tensor] = None,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
    
    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses, **kwargs):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, {"weights": z, "logits": self.w.detach().clone()}

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss      - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()


class LinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(losses * self.task_weights)
        return loss, dict(weights=self.task_weights)


class ScaleInvariantLinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(torch.log(losses) * self.task_weights)
        return loss, dict(weights=self.task_weights)


class STL(WeightMethod):
    """Single task learning"""

    def __init__(self, n_tasks, device: torch.device, main_task):
        super().__init__(n_tasks, device=device)
        self.main_task = main_task
        self.weights = torch.zeros(n_tasks, device=device)
        self.weights[main_task] = 1.0

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        loss = losses[self.main_task]

        return loss, dict(weights=self.weights)


class Uncertainty(WeightMethod):
    """Implementation of `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics`
    Source: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
    """

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)
        self.logsigma = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        loss = sum(
            [
                0.5 * (torch.exp(-logs) * loss + logs)
                for loss, logs in zip(losses, self.logsigma)
            ]
        )

        return loss, dict(
            weights=torch.exp(-self.logsigma)
        )  # NOTE: not exactly task weights

    def parameters(self) -> List[torch.Tensor]:
        return [self.logsigma]




class RLW(WeightMethod):
    """Random loss weighting: https://arxiv.org/pdf/2111.10603.pdf"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        weight = (F.softmax(torch.randn(self.n_tasks), dim=-1)).to(self.device)
        loss = torch.sum(losses * weight)

        return loss, dict(weights=weight)


class DynamicWeightAverage(WeightMethod):
    """Dynamic Weight Average from `End-to-End Multi-Task Learning with Attention`.
    Modification of: https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_split.py#L242
    """

    def __init__(
        self, n_tasks, device: torch.device, iteration_window: int = 25, temp=2.0
    ):
        """

        Parameters
        ----------
        n_tasks :
        iteration_window : 'iteration' loss is averaged over the last 'iteration_window' losses
        temp :
        """
        super().__init__(n_tasks, device=device)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, n_tasks), dtype=np.float32)
        self.weights = np.ones(n_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses, **kwargs):

        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window :, :].mean(0) / self.costs[
                : self.iteration_window, :
            ].mean(0)
            self.weights = (self.n_tasks * np.exp(ws / self.temp)) / (
                np.exp(ws / self.temp)
            ).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(
            losses.device
        )
        loss = (task_weights * losses).mean()

        self.running_iterations += 1

        return loss, dict(weights=task_weights)


class WeightMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        """
        :param method:
        """
        assert method in list(METHODS.keys()), f"unknown method {method}."

        self.method = METHODS[method](n_tasks=n_tasks, device=device, **kwargs)

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)

    def backward(
        self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward(losses, **kwargs)

    def __ceil__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.method.parameters()


METHODS = dict(
    stl=STL,
    ls=LinearScalarization,
    uw=Uncertainty,
    scaleinvls=ScaleInvariantLinearScalarization,
    rlw=RLW,
    dwa=DynamicWeightAverage,

    famo=FAMO,
)
