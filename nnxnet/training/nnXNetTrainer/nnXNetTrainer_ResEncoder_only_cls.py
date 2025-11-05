import os
import pydoc
import csv
import torch
import pydoc
import numpy as np
from time import time, sleep
from torch import autocast, nn
from torch import distributed as dist
from typing import Union, Tuple, List
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule

from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoder_only_cls import ResEncoder_only_cls
from nnxnet.training.nnXNetTrainer.nnXNetTrainer import nnXNetTrainer
from nnxnet.training.dataloading.data_loader_3d_with_global_cls_for_RSNA2025 import nnXNetDataLoader3DWithGlobalCls
from nnxnet.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnxnet.configuration import ANISO_THRESHOLD, default_num_processes
from nnxnet.training.logging.nnxnet_logger import nnXNetLogger
from nnxnet.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnxnet.utilities.collate_outputs import collate_outputs
from nnxnet.utilities.helpers import empty_cache, dummy_context
from nnxnet.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from sklearn.metrics import accuracy_score, roc_auc_score

class nnXNetTrainer_ResEncoder_only_cls(nnXNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)

        self.cls_task_index = self.configuration_manager.cls_task_index
        self.cls_head_num_classes_list = self.configuration_manager.network_arch_init_kwargs["cls_head_num_classes_list"]
        self.print_to_log_file("cls_task_index: ", self.cls_task_index)

        self.num_cls_task = len(self.cls_task_index)

        pos_weights_list = self.configuration_manager.pos_weights_list

        if pos_weights_list is None:
            self.print_to_log_file("pos_weights_list is None, automatically generating pos_weights based on cls_task_index.")
            self.pos_weights_list = [[1.0] * len(cls_labels) if isinstance(cls_labels[0], list) else [1.0] for cls_labels in self.cls_task_index]
        else:
            self.pos_weights_list = pos_weights_list
        
        self.print_to_log_file(f"Using pos_weights_list: {self.pos_weights_list}")

        self.logger = nnXNetLogger(num_cls_task=self.num_cls_task)

        self.save_every = 5
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.cls_loss_list = []
            for i in range(self.num_cls_task):
                pos_weights_tensor = torch.tensor(self.pos_weights_list[i]).to(self.device)
                cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor, reduction='none')
                self.cls_loss_list.append(cls_loss)

            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int) -> nn.Module:

        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
        network = ResEncoder_only_cls(
                in_channels=num_input_channels,
                **architecture_kwargs
            )
        
        return network
    
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = {
                    'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'y': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'z': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                } # TODO: need to revisit 15 degree limit
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnxnet for now)
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            *rotation_for_DA.values(),
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        data = data.to(self.device, non_blocking=True)

        properties = batch['properties']
        keys = batch['keys']
        cls_task1 = torch.from_numpy(batch['cls_task1']).unsqueeze(1).float().to(self.device, non_blocking=True)
        cls_task2 = torch.from_numpy(batch['cls_task2']).float().to(self.device, non_blocking=True)
        cls_task_list = [cls_task1, cls_task2]

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            cls_pred_list = self.network(data)

            total_loss = 0
            total_cls_loss = 0

            for t_index in range(self.num_cls_task):

                cls_pred_logits = cls_pred_list[t_index]

                cls_target = cls_task_list[t_index]
                
                cls_loss = self.cls_loss_list[t_index](cls_pred_logits, cls_target)
                total_loss += cls_loss.mean()
                total_cls_loss += cls_loss.mean()

            l = total_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy(), 'total_cls_loss': total_cls_loss.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        data = data.to(self.device, non_blocking=True)

        properties = batch['properties']
        keys = batch['keys']
        
        cls_task1 = torch.from_numpy(batch['cls_task1']).unsqueeze(1).float().to(self.device, non_blocking=True)
        cls_task2 = torch.from_numpy(batch['cls_task2']).float().to(self.device, non_blocking=True)
        cls_task_list = [cls_task1, cls_task2]

        validation_dict = {}

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            cls_pred_list = self.network(data)

            total_loss = 0
            total_cls_loss = 0

            for t_index in range(self.num_cls_task):

                cls_pred_logits = cls_pred_list[t_index]

                cls_target = cls_task_list[t_index]
                
                cls_loss = self.cls_loss_list[t_index](cls_pred_logits, cls_target)
                total_loss += cls_loss.mean()
                total_cls_loss += cls_loss.mean()

            l = total_loss

        validation_dict['loss'] = l.detach().cpu().numpy()
        validation_dict['total_cls_loss'] = total_cls_loss.detach().cpu().numpy()

        validation_dict['keys'] = keys

        return validation_dict

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            total_cls_train_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            dist.all_gather_object(total_cls_train_losses_tr, outputs['total_cls_loss'])
            loss_here = np.vstack(losses_tr).mean()
            total_cls_loss_here = np.vstack(total_cls_train_losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])
            total_cls_loss_here = np.mean(outputs['total_cls_loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

        if 'total_cls_train_losses' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['total_cls_train_losses'] = list()
        self.logger.log('total_cls_train_losses', total_cls_loss_here, self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        keys = outputs_collated['keys']

        if self.is_ddp:
            world_size = dist.get_world_size()
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()

            total_cls_losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(total_cls_losses_val, outputs_collated['total_cls_loss'])
            total_cls_losses_here = np.vstack(total_cls_losses_val).mean()

            cls_losses_mean = []
            for t_index in range(self.num_cls_task):
                cls_task_losses = [None for _ in range(world_size)]
                dist.all_gather_object(cls_task_losses, [output[f'cls_task_{t_index}_loss'] for output in val_outputs])
                cls_losses_mean.append(np.mean([np.mean(losses) for losses in cls_task_losses]))
        else:
            loss_here = np.mean(outputs_collated['loss'])
            total_cls_losses_here = np.mean(outputs_collated['total_cls_loss'])
            cls_losses_mean = [np.mean([output[f'cls_task_{t_index}_loss'] for output in val_outputs]) 
                            for t_index in range(self.num_cls_task)]
        
        # Prepare CSV data
        csv_data = []
        header = ['Epoch', 'Key', 'Task', 'Loss', 'Classification_Prob', 'Ground_Truth', 'Accuracy', 'AUC']

        for t_index in range(self.num_cls_task):
            self.logger.log(f'cls_task_{t_index}_loss', cls_losses_mean[t_index], self.current_epoch)
            
            cls_probs_list = [output[f'cls_task_{t_index}_probs'] for output in val_outputs]
            cls_targets_list = [output[f'cls_task_{t_index}_targets'] for output in val_outputs]
            
            if self.is_ddp:
                world_size = dist.get_world_size()
                all_cls_probs = [[] for _ in range(world_size)]
                all_cls_targets = [[] for _ in range(world_size)]
                dist.all_gather_object(all_cls_probs, cls_probs_list)
                dist.all_gather_object(all_cls_targets, cls_targets_list)
                cls_probs = np.concatenate(sum(all_cls_probs, []))
                cls_targets = np.concatenate(sum(all_cls_targets, []))
            else:
                cls_probs = np.concatenate(cls_probs_list)
                cls_targets = np.concatenate(cls_targets_list)

            if cls_targets.ndim == 1:
                cls_targets = cls_targets.reshape(-1, 1)
                cls_probs = cls_probs.reshape(-1, 1)

            auc_list = []
            for i in range(cls_probs.shape[1]):
                if len(np.unique(cls_targets[:, i])) > 1:
                    auc_list.append(roc_auc_score(cls_targets[:, i], cls_probs[:, i]))
                else:
                    auc_list.append(0.5)

            auc = np.mean(auc_list)
            
            cls_preds = (cls_probs > 0.5).astype(int)
            acc = accuracy_score(cls_targets.flatten(), cls_preds.flatten())

            self.logger.log(f'cls_task_{t_index}_acc', acc, self.current_epoch)
            self.logger.log(f'cls_task_{t_index}_auc', auc, self.current_epoch)

            # Add data for CSV, ensuring each key is in a separate row
            for key_idx, key in enumerate(keys):
                for sample_idx in range(len(cls_probs[key_idx])):
                    csv_data.append([
                        self.current_epoch,
                        key,
                        f'task_{t_index}',
                        cls_losses_mean[t_index],
                        cls_probs[key_idx][sample_idx],
                        cls_targets[key_idx][sample_idx],
                        acc,
                        auc
                    ])

        self.logger.log('val_losses', loss_here, self.current_epoch)
        if 'total_cls_val_losses' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['total_cls_val_losses'] = list()
        self.logger.log('total_cls_val_losses', total_cls_losses_here, self.current_epoch)

        # Write to CSV file (append mode)
        csv_filename = self.output_folder + '/validation_metrics.csv'
        write_header = not os.path.exists(csv_filename)
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(header)
            writer.writerows(csv_data)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('total_cls_train_losses', np.round(self.logger.my_fantastic_logging['total_cls_train_losses'][-1], decimals=4))
        self.print_to_log_file('total_cls_val_losses', np.round(self.logger.my_fantastic_logging['total_cls_val_losses'][-1], decimals=4))

        for task_i in range(self.num_cls_task):
            self.print_to_log_file(f'cls_task_{task_i}_loss', np.round(self.logger.my_fantastic_logging[f'cls_task_{task_i}_loss'][-1], decimals=4))
            self.print_to_log_file(f'cls_task_{task_i}_acc', np.round(self.logger.my_fantastic_logging[f'cls_task_{task_i}_acc'][-1], decimals=4))
            self.print_to_log_file(f'cls_task_{task_i}_auc', np.round(self.logger.my_fantastic_logging[f'cls_task_{task_i}_auc'][-1], decimals=4))

        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if self._best_ema is None or self.logger.my_fantastic_logging['ema_auc'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_auc'][-1]
            self.print_to_log_file(f"Yayy! New best EMA AUC: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
    
    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            model_dict = self.network.state_dict()
        
            matched_state_dict = {k: v for k, v in new_state_dict.items()
                                if k in model_dict and model_dict[k].shape == v.shape}
            
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(matched_state_dict, strict=False)
            else:
                self.network.load_state_dict(matched_state_dict, strict=False)
        # self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        # if self.grad_scaler is not None:
        #     if checkpoint['grad_scaler_state'] is not None:
        #         self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
