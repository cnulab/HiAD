# Custom Detector
  
The `hiad.detectors.base.BaseDetector` class serves as a wrapper for detectors in HiAD. You can create a custom detector by inheriting from this class.  
  
  
Initialize the detector
```
from hiad.detectors.base import BaseDetector
from typing import List, Union
import torch
import logging

class CustomizedDetector(BaseDetector):

    def __init__(self,
                 other_params,
                 patch_size: Union[int, List],  
                 device: torch.device,  
                 fusion_weights = None,
                 logger: logging.Logger = None, 
                 seed: int = 0,  
                 early_stop_epochs = -1,
                 **kwargs):
        r"""
           Args:
               other_params: Define the parameters required by the anomaly detection algorithm; they should match those provided in the config.
               patch_size (int or list): Image patch resolution.
               device (torch.Device): Computing device in PyTorch.
               fusion_weights (list): Fusion weights used for multi-resolution feature fusion. This functionality is implemented in `BaseDetector`, so you only need to pass the weights to `BaseDetector`.
               logger (logging.Logger): Logger object
               seed (int): random seed.
               early_stop_epochs (int): Controls early stopping during training: 
                        if the detector shows no performance improvement over N epochs, 
                        training will stop early. If set to -1, early stopping is disabled. 
                        This functionality is implemented in `BaseDetector`, so you only need to pass the weights to `BaseDetector`.
        """
        super().__init__(patch_size, device, fusion_weights, logger, seed, early_stop_epochs)
        pass
```

Define the data preprocessing pipeline by implementing the `create_dataset` method.
```
    from hiad.datasets.patch_dataset import PatchDataset
    
    def create_dataset(self, patches: List[LRPatch], training: bool, task_name: str):
        r"""
           This method defines the data preprocessing pipeline for the model by constructing a `Dataset` object.
           Args
               patches (List[LRPatch]): A list of `LRPatch` objects. HiAD encapsulates image patches using the `hiad.utils.split_and_gather.LRPatch` class.
                        Some key attributes included in the `LRPatch` class are:
                        image: np.ndarray               # image patch
                        mask: np.ndarray = None         # Anomaly mask, not `None` only when the patch comes from an anomaly sample used for testing.
                        foreground: np.ndarray = None   # Foreground segmentation mask, typically not used.
                        label: int = None               # 0: normal sample; 1：anomaly sample.
                        label_name: str = None          # The specific anomaly category.
                        clsname: str = None             # Image category.
               training (boolean): True: Trainint mode; False: Inference mode.
               task_name (str):    Task Name
           return:
               return a torch.utils.data.Dataset object
        """
        # The default data loading method provided by HiAD, when overriding this method, refer to the `PatchDataset` class and ensure it returns data in the required format.
        dataset = PatchDataset(patches = patches, training = training, task_name = task_name)
        return dataset
```

Define the feature extraction process by implementing the `embedding` method.
```
    @abstractmethod
    def embedding(self, input_tensor: torch.Tensor ) -> List[torch.Tensor]:
        r"""
           This method encodes image patches into features (feature extraction).
           Args
               input_tensor (torch.Tensor): image patch tensor. Shape: (B,3,Hp,Wp)
           return:
                returns the extracted multi-scale features; Shape: ([B,C1,H1,W1], [B,C2,H2,W2], ..., [B,Cn,Hn,Wn])
        """
        input_tensor = input_tensor.to(self.device)
        # feature extraction process
        raise NotImplementedError
```
Define the model's training process
```
    @abstractmethod
    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str,
                   checkpoint_path: str,
                   val_dataloader: DataLoader = None,
                   evaluators=None,
                   ) -> bool:
        r"""
           This method defines the model's training process.
           Args
               train_dataloader (torch.utils.data.DataLoader): Dataloader used for training; the format of the returned data is defined by the `create_dataset` method.
               task_name (str): Task Name
               checkpoint_path: Path of checkpoint
               val_dataloader (torch.utils.data.DataLoader): Dataloader used for validation; the format of the returned data is defined by the `create_dataset` method.
                    If no val_config is provided, this value will be `None`.
               evaluators: Evaluation methods.
               The `checkpoint_path`, `val_dataloader`, and `evaluators` are used during the model validation and checkpoint saving. 
               The `val_step` method in `BaseDetector` already provides a basic implementation of the validation and checkpoint-saving process, 
               which can be directly invoked using the following code:
                    best_metrics = {}
                    best_metrics = self.val_step(val_dataloader, evaluators, checkpoint_path, best_metrics)
               `best_metrics` Contains the best validation metrics recorded so far.
               Once the `embedding` method is implemented, you can directly obtain multi-resolution fusion features using the `get_multi_resolution_fusion_embeddings` method provided in `BaseDetector`.
               Use multi-resolution fused features instead of the original pre-trained features for anomaly detection:
               for data in train_dataloader:
                   features = self.get_multi_resolution_fusion_embeddings(data)
                   # The shape of `features` is the same as that of the original features. ([B,C1,H1,W1], [B,C2,H2,W2], ..., [B,Cn,Hn,Wn])
           return:
                Returns a boolean: `True` if the checkpoint has already been saved; `False` if not, which will trigger the `trainer` to perform the save.  

        """
        raise NotImplementedError
```
Define the model's inference process
```
    def inference_step(self,
                       test_dataloader: DataLoader,
                       task_name: str):
        r"""
           This method defines the model's inference process.
           Args
               test_dataloader (torch.utils.data.DataLoader): Dataloader used for inference; the format of the returned data is defined by the `create_dataset` method.
               task_name (str): Task Name
    
           return:
               return a list of numpy.ndarray. Each `ndarray` corresponds to the pixel-level detection result of a single patch.
               Shape: ([Hp,Wp],..., [Hp,Wp]).
        """
        raise NotImplementedError
```
Saving and loading of the `checkpoint`.
```
    @abstractmethod
    def save_checkpoint(self,
                        checkpoint_path: str
                        ):
        r"""
            save checkpoint
            Args
                checkpoint_path (str): Path of checkpoint
        """
        raise NotImplementedError


    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        r"""
            load checkpoint
            Args
                checkpoint_path (str): Path of checkpoint
        """
        raise NotImplementedError
```
*Note: All steps described above operate on low-resolution image patches.* 
  
The `to_device` function is used to define the detector's behavior when switching computing devices.
```
    @abstractmethod
    def to_device(self, device: torch.device):
       r"""
            for example: self.model = self.model.to(device)
        """
        raise NotImplementedError
```
You can redefine the computation of image-level anomaly scores by overriding the `get_image_score` method:

```
    @staticmethod
    def get_image_score(segmentations):
        r"""
            This method defines the process for obtaining image-level anomaly detection scores.
            Args
                 segmentations (torch.Tensor): Pixel-level detection results of the high-resolution image, with all patch-level results already concatenate together.
                 Shape: [B,H,W], Device: cpu
            return:
                return image-level anomaly detection scores. Shape: [B]
        """
        pass
```
**If you would like HiAD to support additional detectors, feel free to create a new `issue`. We’ll get back to you as soon as possible!**



