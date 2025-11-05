import os
import pandas as pd
import numpy as np
from nnxnet.training.dataloading.base_data_loader import nnXNetDataLoaderBase
from nnxnet.training.dataloading.nnxnet_dataset import nnXNetDataset

class nnXNetDataLoader3DWithGlobalCls(nnXNetDataLoaderBase):
    def __init__(self, *args, use_sampling_weight=None, **kwargs):
        """
        Parameters:
        use_sampling_weight: Option for using weights
            - None: No weights used, uniform sampling
            - 'modality': Use only modality weights
            - 'vessel': Use only vessel anatomy category weights
        """
        super().__init__(*args, **kwargs)
        
        self.case_sampling_weight_path = os.path.join("/yinghepool/shipengcheng/Dataset/nnUNet/nnUNet_raw/Dataset303_PDCAD", 'train_case_sampling_weight.csv')
        self.csv_path = os.path.join("/yinghepool/shipengcheng/Dataset/nnUNet/nnUNet_raw/Dataset303_PDCAD", 'cls_data.csv')
        
        self.use_sampling_weight = None  #TODO
        
        self.df_full = pd.read_csv(self.csv_path)

        self.identifier_column = 'identifier'
        
        self.cls_columns = ['label']
    
    def get_indices(self):
        return np.random.choice(self.indices, self.batch_size, replace=True, p=self.sampling_probabilities)
    
    def generate_train_batch(self):
        # 1. Select the unique keys (identifiers) for the current batch.
        selected_keys = self.get_indices()

        # 2. Preallocate memory for the main 3D data and segmentation.
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        
        # Preallocate memory for the 1D classification and metadata tasks.
        cls_all = np.zeros((self.batch_size, len(self.cls_columns)), dtype=np.int32)

        for j, i in enumerate(selected_keys):
            # oversampling foreground logic: determines if the patch selection should be biased
            # to include foreground pixels/labels (e.g., to sample more lesions).
            force_fg = self.get_do_oversample(j)

            # Load the 3D data, segmentation, and properties for the current case 'i'.
            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # --- Extract Metadata from CSV for Classification Tasks ---
            sample_data = self.df_full[self.df_full[self.identifier_column] == i].iloc[0]

            for k, cls_col in enumerate(self.cls_columns):
                cls_all[j, k] = int(sample_data[cls_col])
                
            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnXNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        return {
            'data': data_all, 
            'seg': seg_all, 
            'properties': case_properties, 
            'keys': selected_keys,
            'cls_all': cls_all,
        }

if __name__ == '__main__':
    folder = '/yinghepool/shipengcheng/Dataset/nnUNet/nnXNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnXNetDataset(folder, 0)  # this should not load the properties!
    dl = nnXNetDataLoader3DWithGlobalCls(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)