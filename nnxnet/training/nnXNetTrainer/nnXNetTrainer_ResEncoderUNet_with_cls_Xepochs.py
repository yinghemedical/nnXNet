import torch

from nnxnet.training.nnXNetTrainer.nnXNetTrainer_ResEncoderUNet_with_cls import nnXNetTrainer_ResEncoderUNet_with_cls

class nnXNetTrainer_ResEncoderUNet_with_cls_5epochs(nnXNetTrainer_ResEncoderUNet_with_cls):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 5


class nnXNetTrainer_ResEncoderUNet_with_cls_1epoch(nnXNetTrainer_ResEncoderUNet_with_cls):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1


class nnXNetTrainer_ResEncoderUNet_with_cls_10epochs(nnXNetTrainer_ResEncoderUNet_with_cls):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 10


class nnXNetTrainer_ResEncoderUNet_with_cls_20epochs(nnXNetTrainer_ResEncoderUNet_with_cls):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 20


class nnXNetTrainer_ResEncoderUNet_with_cls_50epochs(nnXNetTrainer_ResEncoderUNet_with_cls):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 50


class nnXNetTrainer_ResEncoderUNet_with_cls_100epochs(nnXNetTrainer_ResEncoderUNet_with_cls):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100

class nnXNetTrainer_ResEncoderUNet_with_cls_200epochs(nnXNetTrainer_ResEncoderUNet_with_cls):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 200

class nnXNetTrainer_ResEncoderUNet_with_cls_250epochs(nnXNetTrainer_ResEncoderUNet_with_cls):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250

class nnXNetTrainer_ResEncoderUNet_with_cls_500epochs(nnXNetTrainer_ResEncoderUNet_with_cls):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500

class nnXNetTrainer_ResEncoderUNet_with_cls_2000epochs(nnXNetTrainer_ResEncoderUNet_with_cls):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 2000

    
class nnXNetTrainer_ResEncoderUNet_with_cls_4000epochs(nnXNetTrainer_ResEncoderUNet_with_cls):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 4000


class nnXNetTrainer_ResEncoderUNet_with_cls_8000epochs(nnXNetTrainer_ResEncoderUNet_with_cls):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 8000