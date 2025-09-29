# Advanced Settings
  
This tutorial will further introduce the advanced settings of HiAD to help you understand more of its features.  
  
### Detectors
We have integrated nine anomaly detection methods as detectors in HiAD, which are:  

&nbsp;&nbsp;&nbsp;&nbsp;[PatchCore](https://github.com/amazon-science/patchcore-inspection)  
&nbsp;&nbsp;&nbsp;&nbsp;[PaDiM](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)  
&nbsp;&nbsp;&nbsp;&nbsp;[FastFlow](https://github.com/gathierry/FastFlow)  
&nbsp;&nbsp;&nbsp;&nbsp;[ViTAD](https://github.com/zhangzjn/ader)  
&nbsp;&nbsp;&nbsp;&nbsp;[RealNet](https://github.com/cnulab/RealNet)  
&nbsp;&nbsp;&nbsp;&nbsp;[RD++](https://github.com/tientrandinh/Revisiting-Reverse-Distillation)  
&nbsp;&nbsp;&nbsp;&nbsp;[DeSTSeg](https://github.com/apple/ml-destseg)  
&nbsp;&nbsp;&nbsp;&nbsp;[Dinomaly](https://github.com/guojiajeremy/Dinomaly)  
&nbsp;&nbsp;&nbsp;&nbsp;[INP-Former](https://github.com/luow23/INP-Former)   
  
They are located in the `hiad.detectors` package and can be imported using the following command:  
```
from hiad.detectors import HRPatchCore # PatchCore
from hiad.detectors import HRPaDiM     # PaDiM
from hiad.detectors import HRFastFlow  # FastFlow
from hiad.detectors import HRVitAD     # ViTAD
from hiad.detectors import HRRealNet   # RealNet
from hiad.detectors import HRRDPP      # RD++
from hiad.detectors import HRDesTSeg   # DesTSeg
```
We provide default hyperparameter settings for these methods, which are located in the [configs](../configs) folder and can be loaded using the following command:  

```
from easydict import EasyDict
import yaml

config_path = 'configs/patchcore.yaml' # Take PatchCore as an example
with open(config_path) as f:
    patch_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
config = EasyDict(patch=patch_config)
```
Pass the `detector's class template` and `config` to `HRTrainer` to use the anomaly detection method:
```
from hiad.trainer import HRTrainer
trainer = HRTrainer(detector_class = HRPatchCore, config = config)
```
If you want to add a new anomaly detection method, please refer to [this tutorial](customized_detectors.md).  

### Image Tiling and Task
HiAD uses the `MultiResolutionHRImageSpliter` to divide high-resolution images into smaller image patches. 
It takes three key parameters:

&nbsp;&nbsp;&nbsp;&nbsp; image_size: Image resolution  
&nbsp;&nbsp;&nbsp;&nbsp; patch_size: Patch size  
&nbsp;&nbsp;&nbsp;&nbsp; stride: Tiling stride; Default `None` means it is equal to `patch_size` (i.e., no overlap between image patches)  

Each parameter accepts either an `int` or a `tuple`:
```
from hiad.utils.split_and_gather import MultiResolutionHRImageSpliter  

indexes = MultiResolutionHRImageSpliter(image_size = 2048, patch_size = 512) 
indexes = MultiResolutionHRImageSpliter(image_size = (2048, 2048), patch_size = (512, 512)) # Equivalent forms
indexes = MultiResolutionHRImageSpliter(image_size = (2048, 1024), patch_size = (512, 256)) # Non-square images are supported
```  
`MultiResolutionHRImageSpliter` returns a set of indexes representing the tiling results, which are further divided into `Tasks`.  

`Task` is a key concept in HiAD. It establishes the correspondence between image patches and detectors. HiAD allows training multiple detectors to detect a single high-resolution image. 
We define the following task assignment strategies in HiAD:  

```
from hiad.task import A2OTaskGenerator    # all 2 one
from hiad.task import O2OTaskGenerator    # one 2 one
from hiad.task import NeighborhoodTaskGenerator
from hiad.task import SpatialClusteringTaskGenerator
from hiad.task import ThresholdClusteringTaskGenerator
from hiad.task import RetrieveTaskGenerator
```
`A2OTaskGenerator` assigns all image patches to a single detector:
```
tasks = A2OTaskGenerator().create_tasks(train_samples, indexes)
PrintTasks(tasks) 
# Output:
Row: 4, Column: 4, Patch Number: 16, Task Number: 1
| task_0 | task_0 | task_0 | task_0 | 
| task_0 | task_0 | task_0 | task_0 | 
| task_0 | task_0 | task_0 | task_0 | 
| task_0 | task_0 | task_0 | task_0 | 
```
`O2OTaskGenerator` creates one detector for each image patch:  
```
tasks = O2OTaskGenerator().create_tasks(train_samples, indexes)
PrintTasks(tasks) 
# Output:
Row: 4, Column: 4, Patch Number: 16, Task Number: 16
|  task_0  |  task_1  |  task_2  |  task_3  | 
|  task_4  |  task_5  |  task_6  |  task_7  | 
|  task_8  |  task_9  |  task_10 |  task_11 | 
|  task_12 |  task_13 |  task_14 |  task_15 | 
```
**Note:** Using more detectors generally improves detection performance. 
However, when the resolution is very high, `O2OTaskGenerator` may require training a large number of detectors, 
which may lead to a significant increase in inference latency.  

`NeighborhoodTaskGenerator` assigns image patches within a neighborhood to a single detector. It takes two key parameters:
  
&nbsp;&nbsp;&nbsp;&nbsp; num_groups_width: The number of neighborhoods along the `width` of the image.   
&nbsp;&nbsp;&nbsp;&nbsp; num_groups_height: The number of neighborhoods along the `height` of the image. 
If `None`, it to the same as `num_groups_width`.  
  
The number of detectors is `num_groups_width × num_groups_height`. For example:  
```
tasks = NeighborhoodTaskGenerator(num_groups_width=2, num_groups_height=2).create_tasks(train_samples, indexes)
PrintTasks(tasks) 
# Output:
Row: 4, Column: 4, Patch Number: 16, Task Number: 4
| task_0 | task_0 | task_1 | task_1 | 
| task_0 | task_0 | task_1 | task_1 | 
| task_2 | task_2 | task_3 | task_3 | 
| task_2 | task_2 | task_3 | task_3 | 

tasks = NeighborhoodTaskGenerator(num_groups_width=3, num_groups_height=1).create_tasks(train_samples, indexes)
PrintTasks(tasks) 
# Output:
Row: 4, Column: 4, Patch Number: 16, Task Number: 3
| task_0 | task_0 | task_1 | task_2 | 
| task_0 | task_0 | task_1 | task_2 | 
| task_0 | task_0 | task_1 | task_2 | 
| task_0 | task_0 | task_1 | task_2 | 
```
This results in the following partitioning:  
<div align=center><img width="400" src="..\assets\NA.png"/></div>
  
`SpatialClusteringTaskGenerator` computes the visual similarity of image patches and assigns visually similar patches to the same detector. 
It takes the following key parameters:  
  
&nbsp;&nbsp;&nbsp;&nbsp; cluster_number:  Number of clusters.  
&nbsp;&nbsp;&nbsp;&nbsp; detector_class:  Detector class.  
&nbsp;&nbsp;&nbsp;&nbsp; detector_config: Detector config.  
&nbsp;&nbsp;&nbsp;&nbsp; batch_size:      Feature extraction batch size.  
&nbsp;&nbsp;&nbsp;&nbsp; device: GPU ids.  
&nbsp;&nbsp;&nbsp;&nbsp; feature_resolution: Feature resolution used for the clustering operation, default is 1.  
   
Taking the `Bottle` class from `MVTec-AD` as an example:  
```
tasks = SpatialClusteringTaskGenerator( cluster_number= 4, 
                                        detector_class = detector_class, 
                                        detector_config = config, device = [0],
                                        batch_size= 16).create_tasks(train_samples, indexes)

PrintTasks(tasks)
# Output:
Row: 4, Column: 4, Patch Number: 16, Task Number: 4
| task_0 | task_2 | task_2 | task_0 | 
| task_3 | task_1 | task_1 | task_3 | 
| task_3 | task_1 | task_1 | task_3 | 
| task_0 | task_2 | task_2 | task_0 | 
```
<div align=center><img width="250" src="..\assets\SCA.jpg"/></div>  
  
`ThresholdClusteringTaskGenerator` has functionality similar to `SpatialClusteringTaskGenerator`,
but it does not require a predefined number of clusters. Instead, it uses a threshold parameter, `cluster_threshold`, image patches with similarity greater than `cluster_threshold` will be grouped into the same cluster.  
  
**Note:** For image sets without spatial alignment, such as the `Screw` class in `MVTec-AD`, `ThresholdClusteringTaskGenerator` and `SpatialClusteringTaskGenerator` may produce invalid clustering results.  
  
`RetrieveTaskGenerator` clusters all image patches into `N` clusters based on visual similarity and saves the cluster centers. During inference, it performs dynamic retrieval. 
It removes the dependency on spatial alignment compared to `SpatialClusteringTaskGenerator`, and shares the same input parameters.  
  
### Integrating Low-Resolution Detection
We observed that detecting large-scale anomalies at high resolution often leads to discontinuous detection or missed detections. This is caused by insufficient receptive field coverage in feature extraction.
To address this issue, HiAD can optionally integrate a low-resolution branch to detect large-scale anomalies. You only need to provide an additional configuration for the low-resolution branch:  
  
```
from hiad.detectors import HRPatchCore
from hiad.trainer import HRTrainer

patch_size = 512
thumbnail_size = 256 # Detection resolution for the low-resolution branch

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
       'thumbnail': { # Enable the low-resolution (thumbnail) branch; if not provide, only high-resolution detection is performed.
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
HiAD aggregates detection results from both the high-resolution and low-resolution branches to comprehensively detect anomaly regions of various sizes.
The figure below illustrates the detection results when using `only the high-resolution branch` vs. `using both branches`:  
  
<div align=center><img width="600" src="..\assets\vis2.png"/></div>
<div align=center><img width="600" src="..\assets\vis3.png"/></div>  

Loaded via the configuration file:
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

### Multi-Resolution Feature Fusion
High-resolution anomaly detection is prone to over-detection (small false positive regions) due to interference from fine-grained texture details in the image.
To address this, HiAD can enhance the model’s robustness to texture variations by `fusing features from different resolutions`. This functionality can be enabled by providing an additional `downsampling factor` and `fusion weight`:  

```
image_size = 2048
patch_size = 512
batch_size = 8

ds_factors = [0, 1]  # Downsampling factor, this controls the fusion of image features at resolutions 2048*2048 and 1024*1024. The calculation is as follows: [2048/2^0, 2048/2^1] = [2048, 1024].
fusion_weights = [0.5, 0.5] # Fusion weights for features at each resolution, len(fusion_weights) == len(ds_factors)

indexes = MultiResolutionHRImageSpliter(image_size = image_size, 
                                        patch_size = patch_size, 
                                        ds_factors = ds_factors  # downsampling factors
                                        )

tasks = SpatialClusteringTaskGenerator( cluster_number= 4, 
                                        detector_class = detector_class, 
                                        detector_config = config, device = [0],
                                        fusion_weights = fusion_weights, # fusion weights
                                        batch_size = batch_size).create_tasks(train_samples, indexes)

trainer = HRTrainer(batch_size = batch_size,
                    tasks = tasks,
                    detector_class = detector_class,
                    config = config,
                    fusion_weights = fusion_weights, # fusion weights
                    checkpoint_root = checkpoints_root,
                    log_root = log_root,
                    vis_root = vis_root)
```
Printing the indexes shows that HiAD has added additional low-resolution indexes for each patch:  
``` 
for index in indexes:
    print(index)
# Output：
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
### Evaluation  
  
HiAD performs model evaluation by passing a set of evaluators to `hiad.HRTrainer.inference`. Evaluation can be conducted on either `GPU` or `CPU`:
```
from functools import partial

# Evaluation using GPU computation
from hiad.utils import metrics_gpu
evaluators = [
      metrics_gpu.compute_imagewise_metrics_gpu, # image-auroc
      partial(metrics_gpu.compute_pixelwise_metrics_gpu_resize, resize = 512), # pixel-auroc, pixel-f1, pixel-ap
      partial(metrics_gpu.compute_pro_gpu_resize, resize = 512), # pro
  ]
  
  
# Evaluation using CPU computation
from hiad.utils import metrics
evaluators = [
      metrics.compute_imagewise_metrics_gpu, # image-auroc
      partial(metrics.compute_pixelwise_metrics_gpu_resize, resize = 512), # pixel-auroc, pixel-f1, pixel-ap
      partial(metrics.compute_pro_gpu_resize, resize = 512), # pro
  ]
```
The `resize` parameter is used to specify the resolution at which pixel-level metrics are computed. We found that computing pixel-level metrics at a suitably downsampled resolution can significantly improve speed,
with minimal impact on accuracy.  

If you want to define a new evaluation metric, you can refer to the following code:  
```
def customized_evaluation_metrics(
    prediction_scores, gt_labels, 
    prediction_masks, gt_masks,
    device = None, **kwargs
):
    """
       Computes customized metrics
       Args:
           prediction_scores: image-level scores predicted by models. shape: [N]
           gt_labels: Ground-truth image-level labels. shape: [N]
           prediction_masks:  pixel-level scores predicted by models. shape: [numpy.ndarray(H*W),...,numpy.ndarray(H*W)]
           gt_masks:  Ground-truth pixel-level masks. shape: [numpy.ndarray(H*W),...,numpy.ndarray(H*W)]
           device: torch.device
    """
    ## Computation process
    return {"metric_1_name": metric_1, ..., "metric_n_name": metric_n}

trainer.inference( evaluators = [customized_evaluation_metrics])
```
### Validation set
  
The performance of some anomaly detection methods is highly dependent on the `model checkpoint selection`.
HiAD performs `checkpoint selection` and `anomaly score normalization` using a constructed `validation set`.
You can create a validation set by passing in a validation config:  
  
```
from hiad.syn import RandomeBoxSynthesizer
from hiad.utils import metrics
from hiad.trainer import HRTrainer

trainer = HRTrainer()

evaluators = [
    metrics.compute_imagewise_metrics,
    metrics.compute_pixelwise_metrics,
] # Define the evaluation method for the validation set. 
  # Note: It is recommended not to compute this on the GPU, as it may lead to additional GPU memory usage during training.


val_config = {
    'anomaly_syn_fn': RandomeBoxSynthesizer(p=0.5), # Anomaly synthesis method used to generate anomaly samples, where `p` represents the probability of anomaly synthesis.
    'val_sample_ratio': 0.2,  # Proportion of training samples used to construct the validation set.  
    'max_patch_number': 200,  # Maximum number of image patches in the validation set; if not specified, means no limit. 
    'evaluators': evaluators  
}
trainer.train(val_config = val_config, **)
```

HiAD integrates the following anomaly synthesis methods:  
```
from hiad.syn import ImageBlendingSynthesizer      # Image blending anomaly synthesis, synthesize anomalies by blending image patches with anomaly source images (e.g., from the DTD dataset).
from hiad.syn import RandomeBoxSynthesizer         # Synthesize anomalies by placing rectangles with random colors on image patches.
from hiad.syn import CutPasteAnomalySynthesizer    # CutPaste anomaly synthesis.
from hiad.syn import ColorShiftSynthesizer         # Apply simple data augmentations to image patches, such as color and brightness variations. 
```
  
If `val_config` is provided during training phase, the anomaly scores output by the `trainer` will be normalized to the range `[0, 1]`.  






