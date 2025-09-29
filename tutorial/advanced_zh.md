# 高级设置
  
本教程将进一步介绍关于HiAD的高级设置，帮助您了解HiAD的更多功能。
  
### 检测器
我们在HiAD中集成了`9种`异常检测方法作为检测器，分别是：  

&nbsp;&nbsp;&nbsp;&nbsp;[PatchCore](https://github.com/amazon-science/patchcore-inspection)  
&nbsp;&nbsp;&nbsp;&nbsp;[PaDiM](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)  
&nbsp;&nbsp;&nbsp;&nbsp;[FastFlow](https://github.com/gathierry/FastFlow)  
&nbsp;&nbsp;&nbsp;&nbsp;[ViTAD](https://github.com/zhangzjn/ader)  
&nbsp;&nbsp;&nbsp;&nbsp;[RealNet](https://github.com/cnulab/RealNet)  
&nbsp;&nbsp;&nbsp;&nbsp;[RD++](https://github.com/tientrandinh/Revisiting-Reverse-Distillation)  
&nbsp;&nbsp;&nbsp;&nbsp;[DeSTSeg](https://github.com/apple/ml-destseg)  
&nbsp;&nbsp;&nbsp;&nbsp;[Dinomaly](https://github.com/guojiajeremy/Dinomaly)  
&nbsp;&nbsp;&nbsp;&nbsp;[INP-Former](https://github.com/luow23/INP-Former)   

它们位于`hiad.detectors`包，可以通过以下命令导入
```
from hiad.detectors import HRPatchCore # PatchCore
from hiad.detectors import HRPaDiM     # PaDiM
from hiad.detectors import HRFastFlow  # FastFlow
from hiad.detectors import HRVitAD     # ViTAD
from hiad.detectors import HRRealNet   # RealNet
from hiad.detectors import HRRDPP      # RD++
from hiad.detectors import HRDesTSeg   # DesTSeg
```
我们为这些方法提供了默认的超参数设置，位于文件夹[configs](../configs)，可以通过以下命令加载：

```
from easydict import EasyDict
import yaml

config_path = 'configs/patchcore.yaml' # 以PatchCore为例
with open(config_path) as f:
    patch_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
config = EasyDict(patch=patch_config)
```
将检测器的类模版和config传递给`HRTrainer`，即可使用该检测算法：
```
from hiad.trainer import HRTrainer
trainer = HRTrainer(detector_class = HRPatchCore, config = config)
```
如果您想添加新的异常检测算法，可以参考[此教程](customized_detectors_zh.md) 。

### 图像分块与Task
HiAD使用`MultiResolutionHRImageSpliter`将高分辨率图像切割为小的图像块，它接收三个关键参数：

&nbsp;&nbsp;&nbsp;&nbsp; image_size: 输入图像分辨率  
&nbsp;&nbsp;&nbsp;&nbsp; patch_size: 划分的图像块大小  
&nbsp;&nbsp;&nbsp;&nbsp; stride: 分块步长，默认为`None`表示与patch_size相等 (图像块之间没有重叠)  

每个参数接收一个int或元组：
```
from hiad.utils.split_and_gather import MultiResolutionHRImageSpliter  

indexes = MultiResolutionHRImageSpliter(image_size = 2048, patch_size = 512) 
indexes = MultiResolutionHRImageSpliter(image_size = (2048, 2048), patch_size = (512, 512)) # 等价写法
indexes = MultiResolutionHRImageSpliter(image_size = (2048, 1024), patch_size = (512, 256)) # 支持非正方形图像
```  
`MultiResolutionHRImageSpliter`返回一组表示分块结果的索引，这些索引将进一步被划分为`Task`。  

Task是HiAD中的一个重要概念，它建立了图像块与检测器之间的对应关系。HiAD可以训练多个检测器来检测一个高分辨率图像。我们在HiAD中定义了以下几种任务分配方案：
```
from hiad.task import A2OTaskGenerator    # all 2 one
from hiad.task import O2OTaskGenerator    # one 2 one
from hiad.task import NeighborhoodTaskGenerator
from hiad.task import SpatialClusteringTaskGenerator
from hiad.task import ThresholdClusteringTaskGenerator
from hiad.task import RetrieveTaskGenerator
```
`A2OTaskGenerator`将所有图像块分配给同一个检测器：
```
tasks = A2OTaskGenerator().create_tasks(train_samples, indexes)
PrintTasks(tasks) 
# 输出：
Row: 4, Column: 4, Patch Number: 16, Task Number: 1
| task_0 | task_0 | task_0 | task_0 | 
| task_0 | task_0 | task_0 | task_0 | 
| task_0 | task_0 | task_0 | task_0 | 
| task_0 | task_0 | task_0 | task_0 | 
```
`O2OTaskGenerator`为每个图像块创建一个检测器：
```
tasks = O2OTaskGenerator().create_tasks(train_samples, indexes)
PrintTasks(tasks) 
# 输出：
Row: 4, Column: 4, Patch Number: 16, Task Number: 16
|  task_0  |  task_1  |  task_2  |  task_3  | 
|  task_4  |  task_5  |  task_6  |  task_7  | 
|  task_8  |  task_9  |  task_10 |  task_11 | 
|  task_12 |  task_13 |  task_14 |  task_15 | 
```
注：使用更多的检测器通常会提高检测性能。但分辨率过高时，`O2OTaskGenerator`可能需要训练非常多的检测器，造成推理时延的显著增加。
  
`NeighborhoodTaskGenerator`将一个邻域内的图像块分配给一个检测器，它接收两个关键参数：  
  
&nbsp;&nbsp;&nbsp;&nbsp; num_groups_width: 将图像按`宽度`方向划分的邻域数量   
&nbsp;&nbsp;&nbsp;&nbsp; num_groups_height: 将图像按`高度`方向划分的邻域数量，缺省时将与num_groups_width相同   
  
检测器数量等于`num_groups_width × num_groups_height`，例如：
```
tasks = NeighborhoodTaskGenerator(num_groups_width=2, num_groups_height=2).create_tasks(train_samples, indexes)
PrintTasks(tasks) 
# 输出：
Row: 4, Column: 4, Patch Number: 16, Task Number: 4
| task_0 | task_0 | task_1 | task_1 | 
| task_0 | task_0 | task_1 | task_1 | 
| task_2 | task_2 | task_3 | task_3 | 
| task_2 | task_2 | task_3 | task_3 | 

tasks = NeighborhoodTaskGenerator(num_groups_width=3, num_groups_height=1).create_tasks(train_samples, indexes)
PrintTasks(tasks) 
# 输出：
Row: 4, Column: 4, Patch Number: 16, Task Number: 3
| task_0 | task_0 | task_1 | task_2 | 
| task_0 | task_0 | task_1 | task_2 | 
| task_0 | task_0 | task_1 | task_2 | 
| task_0 | task_0 | task_1 | task_2 | 
```
这产生以下的划分结果：
<div align=center><img width="400" src="..\assets\NA.png"/></div>
  
`SpatialClusteringTaskGenerator`计算不同位置图像块的视觉相似度，将外观相似的图像块分配给相同的检测器，它接收以下关键参数： 
  
&nbsp;&nbsp;&nbsp;&nbsp; cluster_number: 聚类个数  
&nbsp;&nbsp;&nbsp;&nbsp; detector_class: 检测器类  
&nbsp;&nbsp;&nbsp;&nbsp; detector_config: 检测器配置  
&nbsp;&nbsp;&nbsp;&nbsp; batch_size: 特征提取批处理大小   
&nbsp;&nbsp;&nbsp;&nbsp; device: GPU ids  
&nbsp;&nbsp;&nbsp;&nbsp; feature_resolution: 用于执行聚类操作的特征分辨率，默认为1  
  
以MVTec-AD中的Bottle类为例：
```
tasks = SpatialClusteringTaskGenerator( cluster_number= 4, 
                                        detector_class = detector_class, 
                                        detector_config = config, device = [0],
                                        batch_size= 16).create_tasks(train_samples, indexes)

PrintTasks(tasks)
# 输出：
Row: 4, Column: 4, Patch Number: 16, Task Number: 4
| task_0 | task_2 | task_2 | task_0 | 
| task_3 | task_1 | task_1 | task_3 | 
| task_3 | task_1 | task_1 | task_3 | 
| task_0 | task_2 | task_2 | task_0 | 
```
<div align=center><img width="250" src="..\assets\SCA.jpg"/></div>  

`ThresholdClusteringTaskGenerator`与`SpatialClusteringTaskGenerator`基本功能相同，
但`ThresholdClusteringTaskGenerator`无需预先指定聚类个数，而是通过阈值`cluster_threshold`将相似度大于`cluster_threshold`的图像块划分为一类。  

注：对于未经位置校准的图像集合，例如MVTec-AD中的Screw类，`ThresholdClusteringTaskGenerator`与`SpatialClusteringTaskGenerator`可能产生无效的聚类结果。
  
  
`RetrieveTaskGenerator`将所有图像块按照相似度聚类为N个簇，保存聚类中心，在推理时执行动态检索。在`SpatialClusteringTaskGenerator`的基础上消除了对于图像对齐的依赖，接收的参数与`SpatialClusteringTaskGenerator`相同。

### 集成低分辨率检测
我们观察到，在高分辨率下检查大面积异常容易出现检测不连续或漏检的情况，这是由于特征感受野不足造成的。
为了解决这个问题，HiAD可以选择性的集成一个低分辨率分支来检测大面积异常，仅需额外提供低分辨率分支的参数配置即可：
```
from hiad.detectors import HRPatchCore
from hiad.trainer import HRTrainer

patch_size = 512
thumbnail_size = 256 # 低分辨率分支检测分辨率

config = {
       'patch': {
           'backbone_name': 'wideresnet50',
           'layers_to_extract_from': ['layer2', 'layer3'],
           'merge_size': 3,
           'percentage': 0.1,
           'pretrain_embed_dimension': 1024,
           'target_embed_dimension': 1024,
           'patch_size': patch_size,
       },
       'thumbnail': {    # 启用低分辨率分支（缩略图），缺省时则只进行高分辨率检测
           'backbone_name': 'wideresnet50',
           'layers_to_extract_from': ['layer2', 'layer3'],
           'merge_size': 3,
           'percentage': 0.1,
           'pretrain_embed_dimension': 1024,
           'target_embed_dimension': 1024,
           'thumbnail_size': thumbnail_size,   
       }
   }

trainer = HRTrainer(detector_class = HRPatchCore, config = config)
```
HiAD将汇总高分辨率和低分辨率分支的检测结果，以全面检测各种大小的异常区域。下图展示了`仅使用高分辨率分支`和`同时使用两个分支`的检测结果：
<div align=center><img width="600" src="..\assets\vis2.png"/></div>
<div align=center><img width="600" src="..\assets\vis3.png"/></div>  

通过配置文件加载
```
from easydict import EasyDict
import yaml

thumbnail_size = 256
patch_path = 'configs/fastflow_patch.yaml' 
with open(patch_path) as f:
    patch_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    
thumbnail_path = 'configs/fastflow_thumbnail.yaml' 
with open(thumbnail_path) as f:
    thumbnail_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    
thumbnail_config.thumbnail_size = thumbnail_size
config = EasyDict(patch = patch_config, thumbnail = thumbnail_config)
```

### 多分辨率特征融合
高分辨率异常检测容易受到图像中纹理细节的干扰而产生误检（细小的假阳性区域）。为此，HiAD可以通过融合不同分辨率的特征以提升模型对于纹理变化的鲁棒性，
我们可以通过提供额外的`下采样因子`和`融合权重`来启用此功能：
```
image_size = 2048
patch_size = 512
batch_size = 8

ds_factors = [0, 1]  # 下采样因子，将融合 2048*2048 和 1024*1024 分辨率的图像特征，计算方式如下：[2048/2^0, 2048/2^1] = [2048, 1024]
fusion_weights = [0.5, 0.5] # 各分辨率特征的融合权重，len(fusion_weights) == len(ds_factors)

indexes = MultiResolutionHRImageSpliter(image_size = image_size, 
                                        patch_size = patch_size, 
                                        ds_factors = ds_factors  # 添加下采样因子
                                        )

tasks = SpatialClusteringTaskGenerator( cluster_number= 4, 
                                        detector_class = detector_class, 
                                        detector_config = config, device = [0],
                                        fusion_weights = fusion_weights, # 添加融合权重
                                        batch_size = batch_size).create_tasks(train_samples, indexes)

trainer = HRTrainer(batch_size = batch_size,
                    tasks = tasks,
                    detector_class = detector_class,
                    config = config,
                    fusion_weights = fusion_weights, # 添加融合权重
                    checkpoint_root = checkpoints_root,
                    log_root = log_root,
                    vis_root = vis_root)
```  
此时打印图像块的索引，可以看到HiAD为每个图像块添加了额外的低分辨率索引：
``` 
for index in indexes:
    print(index)
# 输出：
{"main_index": {"x": 0, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 0, "y": 0, "width": 1024, "height": 1024}]}
{"main_index": {"x": 512, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 0, "y": 0, "width": 1024, "height": 1024}]}
{"main_index": {"x": 1024, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 1024, "y": 0, "width": 1024, "height": 1024}]}
{"main_index": {"x": 1536, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 1024, "y": 0, "width": 1024, "height": 1024}]}
{"main_index": {"x": 0, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 0, "y": 0, "width": 1024, "height": 1024}]}
{"main_index": {"x": 512, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 0, "y": 0, "width": 1024, "height": 1024}]}
{"main_index": {"x": 1024, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 1024, "y": 0, "width": 1024, "height": 1024}]}
{"main_index": {"x": 1536, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 1024, "y": 0, "width": 1024, "height": 1024}]}
{"main_index": {"x": 0, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 0, "y": 1024, "width": 1024, "height": 1024}]}
{"main_index": {"x": 512, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 0, "y": 1024, "width": 1024, "height": 1024}]}
{"main_index": {"x": 1024, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 1024, "y": 1024, "width": 1024, "height": 1024}]}
{"main_index": {"x": 1536, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 1024, "y": 1024, "width": 1024, "height": 1024}]}
{"main_index": {"x": 0, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 0, "y": 1024, "width": 1024, "height": 1024}]}
{"main_index": {"x": 512, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 0, "y": 1024, "width": 1024, "height": 1024}]}
{"main_index": {"x": 1024, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 1024, "y": 1024, "width": 1024, "height": 1024}]}
{"main_index": {"x": 1536, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": [{"x": 1024, "y": 1024, "width": 1024, "height": 1024}]}
``` 

### 模型评价
HiAD通过向`hiad.HRTrainer.inference`中传入一组`evaluators`进行模型评估，评估可以在GPU或CPU进行，
它们分别位于`hiad.utils.metrics_gpu`和`hiad.utils.metrics`中：
```
from functools import partial

# 通过GPU计算
from hiad.utils import metrics_gpu
evaluators = [
      metrics_gpu.compute_imagewise_metrics_gpu, # 计算image-auroc
      partial(metrics_gpu.compute_pixelwise_metrics_gpu_resize, resize = 512), # 计算pixel-auroc, pixel-f1, pixel-ap
      partial(metrics_gpu.compute_pro_gpu_resize, resize = 512), # 计算pro
  ]
  
  
# 通过CPU计算
from hiad.utils import metrics
evaluators = [
      metrics.compute_imagewise_metrics_gpu, # 计算image-auroc
      partial(metrics.compute_pixelwise_metrics_gpu_resize, resize = 512), # 计算pixel-auroc, pixel-f1, pixel-ap
      partial(metrics.compute_pro_gpu_resize, resize = 512), # 计算pro
  ]
```
参数`resize`用于指定计算像素级指标的分辨率，我们发现在适当下采样的分辨率上计算像素级指标能显著提升计算速度，且几乎不会产生明显的偏差。
  
如果想定义新的评价指标，可参考以下代码：
```
def customized_evaluation_metrics(
    prediction_scores, gt_labels, 
    prediction_masks, gt_masks,
    device = None, **kwargs
):
    """
       Computes customized metrics
       Args:
           prediction_scores: 模型预测图像级分数。shape: [N]
           gt_labels: 真实图像级标签。shape: [N]
           prediction_masks:  模型预测像素级分数。shape: [numpy.ndarray(H*W),...,numpy.ndarray(H*W)]
           gt_masks:  真实像素级掩码。shape: [numpy.ndarray(H*W),...,numpy.ndarray(H*W)]
           device:    torch.device
    """
    ## 计算过程
    return {"metric_1_name": metric_1, ..., "metric_n_name": metric_n}

trainer.inference( evaluators = [customized_evaluation_metrics])
```
### 验证集
一些异常检测方法的性能高度依赖`模型检查点`的选择，HiAD通过构造的`验证集`进行`模型检查点选择`和`异常分数归一化`，
我们可以通过传入验证集配置来创建验证集：
```
from hiad.syn import RandomeBoxSynthesizer
from hiad.trainer import HRTrainer
from hiad.utils import metrics

trainer = HRTrainer()

evaluators = [
    metrics.compute_imagewise_metrics,
    metrics.compute_pixelwise_metrics,
] # 定义验证集评价方法。注意：最好不要在GPU上计算，这会导致训练期间额外的显存占用。

val_config = {
    'anomaly_syn_fn': RandomeBoxSynthesizer(p=0.5), # 异常合成方法，用于构造验证集中的异常样本，p表示异常合成概率。
    'val_sample_ratio': 0.2,  # 用于构造验证集的训练样本比例
    'max_patch_number': 200,  # 验证集中的最大图像块数量，若缺省则无限制
    'evaluators': evaluators  
}
trainer.train(val_config = val_config, **)
```

HiAD中集成了以下的异常合成方法：
```
from hiad.syn import ImageBlendingSynthesizer      # 图像混合异常合成，通过异常源图像（如DTD数据集）进行异常合成
from hiad.syn import RandomeBoxSynthesizer         # 通过在图像块上放置具有随机颜色的矩形进行异常合成
from hiad.syn import CutPasteAnomalySynthesizer    # CutPaste异常合成
from hiad.syn import ColorShiftSynthesizer         # 对图像块进行简单的数据增强，如颜色、亮度变化
```
  
若在训练过程中提供了`val_config`，则模型输出的异常分数将会被归一化到`[0,1]`之间。








