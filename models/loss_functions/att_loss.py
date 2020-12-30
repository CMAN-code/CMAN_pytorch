import torch

from models.base import BaseModule


class AttLoss(BaseModule):
    """
    Implements the reconstruction loss.
    """
    def __init__(self):
        # type: () -> None
        """
        Class constructor.
        """
        super(AttLoss, self).__init__()

    def forward(self, w):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :return: the mean reconstruction loss (averaged along the batch axis).
        """
        L = - w * torch.log(w + 1e-5)

        while L.dim() > 1:
            L = torch.sum(L, dim=-1)

        return torch.mean(L)
