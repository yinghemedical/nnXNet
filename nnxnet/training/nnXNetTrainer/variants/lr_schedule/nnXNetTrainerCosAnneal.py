import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnxnet.training.nnXNetTrainer.nnXNetTrainer import nnXNetTrainer


class nnXNetTrainerCosAnneal(nnXNetTrainer):
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler

