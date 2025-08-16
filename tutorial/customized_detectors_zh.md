# 自定义检测器

`hiad.detectors.base.BaseDetector`类对HiAD中的检测器进行了封装，可以通过继承该类来创建新的检测器。  
  
初始化检测器
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
               other_params: 定义异常检测算法所需的参数，与提供的config对应即可。
               patch_size (int or list): 输入的图像块分辨率
               device (torch.Device): PyTorch中的计算设备
               fusion_weights (list): 用于多分辨率特征融合的融合权重。其功能在BaseDetector中实现，只需要传递给BaseDetector即可。
               logger (logging.Logger): 日志对象
               seed (int): 随机种子. 
               early_stop_epochs (int): 控制训练的提前停止：如果检测器在N个epoch中没有性能提升，
                    训练会提前停止。如果为-1，将禁用提前停止。其功能在BaseDetector中实现，只需要传递给BaseDetector即可。
        """
        super().__init__(patch_size, device, fusion_weights, logger, seed, early_stop_epochs)
        pass
```

通过实现`create_dataset`方法定义数据预处理流程
```
    from hiad.datasets.patch_dataset import PatchDataset
    
    def create_dataset(self, patches: List[LRPatch], training: bool, task_name: str):
        r"""
           此方法通过构建 `Dataset` 来定义模型数据预处理流程。
           Args
               patches (List[LRPatch]): 一组LRPatch对象列表，HiAD将图像块封装为hiad.utils.split_and_gather.LRPatch类
                    LRPatch类包含的一些核心属性：
                        image: np.ndarray                # 图像块
                        mask: np.ndarray = None          # 异常掩码，仅当用于测试的异常样本时不为None
                        foreground: np.ndarray = None    # 前景分割掩码，通常不会被使用
                        label: int = None                # 0:正常样本；1：异常样本
                        label_name: str = None           # 保存具体的异常类别
                        clsname: str = None              # 保存图像类别
               training (boolean): True: 训练模式；False: 推理模式
               task_name (str):    Task名称
           return:
               返回一个 torch.utils.data.Dataset 对象
        """
        # HiAD提供的默认数据加载方式，重写该方法时请参考PatchDataset类，返回特定的数据格式
        dataset = PatchDataset(patches = patches, training = training, task_name = task_name)
        return dataset
```

通过实现`embedding`方法定义特征提取流程
```
    @abstractmethod
    def embedding(self, input_tensor: torch.Tensor ) -> List[torch.Tensor]:
        r"""
           此方法将图像块编码为特征（特征提取）。
           Args
               input_tensor (torch.Tensor): 输入的图像块tensor. Shape: (B,3,Hp,Wp)
           return:
                返回提取的多尺度特征 Shape: ([B,C1,H1,W1], [B,C2,H2,W2], ..., [B,Cn,Hn,Wn])
        """
        input_tensor = input_tensor.to(self.device)
        # 特征提取过程
        raise NotImplementedError
```
定义模型的训练过程
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
           该方法定义了模型的训练过程
           
           Args
               train_dataloader (torch.utils.data.DataLoader): 用于训练的Dataloader；返回数据的格式由 `create_dataset`方法定义。
               task_name (str): 任务名称
               checkpoint_path: 模型检查点保存路径
               val_dataloader (torch.utils.data.DataLoader): 用于验证的Dataloader；返回数据的格式由 `create_dataset`方法定义。
                    如果没有提供验证配置，该值将为None。
               evaluators: 评估方法
               其中checkpoint_path、val_dataloader和evaluators都用于模型验证和检查点保存。BaseDetector中的val_step方法已经对模型验证
               和检查点保存流程进行了基础的实现，可以通过以下代码直接调用：
                    best_metrics = {}
                    best_metrics = self.val_step(val_dataloader, evaluators, checkpoint_path, best_metrics)
               `best_metrics` 包含历史最佳验证指标
               当实现了embedding方法，可以通过BaseDetector中的方法get_multi_resolution_fusion_embeddings方法直接获得多分辨率融合特征，
               使用多分辨率融合特征替代原始预训练特征进行异常检测：
               for data in train_dataloader:
                   features = self.get_multi_resolution_fusion_embeddings(data)
                   # features的shape与原始特征相同 ([B,C1,H1,W1], [B,C2,H2,W2], ..., [B,Cn,Hn,Wn])
    
           return:
                返回一个boolean： 如果已保存检查点，则返回 True；如果未保存，则返回 False，这将触发 `trainer` 执行保存。
        """
        raise NotImplementedError
```
定义模型的推理过程
```
    def inference_step(self,
                       test_dataloader: DataLoader,
                       task_name: str):
        r"""
           该方法定义了模型的推理过程
    
           Args
               test_dataloader (torch.utils.data.DataLoader): 用于推理的Dataloader；返回数据的格式由 `create_dataset`方法定义。
               task_name (str): 任务名称
    
           return:
               返回一个 numpy.ndarray 列表，其中每个ndarray对应单个图像块的像素级检测结果。
               Shape: ([Hp,Wp],..., [Hp,Wp]).
        """
        raise NotImplementedError
```
定义`checkpoint`的保存和加载
```
    @abstractmethod
    def save_checkpoint(self,
                        checkpoint_path: str
                        ):
        r"""
            保存checkpoint
            Args
                checkpoint_path (str): checkpoint路径
        """
        raise NotImplementedError


    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        r"""
            加载checkpoint
            Args
                checkpoint_path (str): checkpoint路径
        """
        raise NotImplementedError
```
*注：上述流程中的处理对象均为切割后的图像块。*  
  
通过重写`get_image_score`方法可以重新定义图像级检测分数的计算过程：
```
    @staticmethod
    def get_image_score(segmentations):
        r"""
            此方法定义了图像级异常检测分数的获取过程
            Args
                 segmentations (torch.Tensor): 高分辨率图像的像素级级检测结果，此时已将检测结果进行了拼接.
                 Shape: [B,H,W], Device: cpu
            return:
                返回图像级分数. Shape: [B]
        """
        pass
```
**当然！如果你希望HiAD能够支持其他的检测器，可以创建新的`issues`，我们将尽快回复！**

