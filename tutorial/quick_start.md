# Quick Start
  
This tutorial provides a simple example to help you quickly understand how HiAD works. Running it only requires an 8GB GPU and about 10 minutes.
### Dataset
First, we have prepared a mini-dataset for demonstration purposes. It is constructed from the 'leather'
class of the MVTec-AD dataset using super-resolution and defect synthesis techniques, with an image 
resolution of `2048 × 2048`. We use this mini-dataset to simulate a high-resolution anomaly detection 
task. The dataset can be downloaded from the following link:  
[data.zip (13MB)](data.zip)  
  

### Installation
Install HiAD using the following command:
```
$ pip install hiad[cuda11] # for Linux with cuda11 
$ pip install hiad[cuda12] # for Linux with cuda12
$ pip install hiad[cuda]   # for Linux with other cuda versions
$ pip install hiad         # for Windows
```  
<sub><em>Since `faiss-gpu` is not supported on Windows, some features of HiAD may be limited on Windows systems.</em></sub>

### Training and Inference
HiAD can extend existing anomaly detection methods to arbitrarily high-resolution scenarios by `splitting high-resolution images into smaller patches for individual detection and then aggregating the results`.

First, we define some basic variables:

```
import os 

image_size = 2048 # Detection Resolution
patch_size = 512  # Size of Image Patches
batch_size = 2    # Batch Size
gpus = [0]        # GPU IDs; using multiple GPUs will significantly improve training and inference speed.

data_root = 'data'   # Extraction directory of the dataset data.zip
log_root = 'logs'    # Path to save log files
checkpoints_root = 'saved_models' # Model checkpoint path
vis_root = 'vis'     # Path to save visualization results

os.makedirs(log_root, exist_ok=True)
os.makedirs(checkpoints_root, exist_ok=True)
os.makedirs(vis_root, exist_ok=True)
```
  
In this example, we use the `PatchCore` algorithm integrated within HiAD for anomaly detection:  
```
from hiad.detectors import HRPatchCore  # Import the algorithm template class

detector_class = HRPatchCore 
config = { 
      'patch': {
          'backbone_name': 'wideresnet50',
          'layers_to_extract_from': ['layer2', 'layer3'],
          'merge_size': 3,
          'percentage': 0.1,
          'pretrain_embed_dimension': 1024,
          'target_embed_dimension': 1024,
          'patch_size': patch_size,
      }
  } # Set hyperparameters for PatchCore
```
  
Load samples for training and testing:
```
train_image_paths = [os.path.join(data_root, file) for file in os.listdir(data_root) if file.startswith('train')]
print(train_image_paths)
test_image_paths = [os.path.join(data_root, file) for file in os.listdir(data_root) if file.startswith('test') and file.find('mask')==-1]
print(test_image_paths)
```
The dataset contains 10 training images and 10 testing images (5 normal samples + 5 anomalous samples).  
``` 
['data/train000.jpg', 'data/train003.jpg', 'data/train004.jpg', 'data/train005.jpg', 'data/train006.jpg', 'data/train008.jpg', 'data/train009.jpg', 'data/train001.jpg', 'data/train002.jpg', 'data/train007.jpg']
['data/test004.jpg', 'data/test000.jpg', 'data/test001.jpg', 'data/test002.jpg', 'data/test003.jpg', 'data/test005.jpg', 'data/test008.jpg', 'data/test009.jpg', 'data/test006.jpg', 'data/test007.jpg']
``` 
HiAD encapsulates high-resolution images into the `HRSample` class, which is initialized using the image path:  
```     
from hiad.utils.split_and_gather import HRSample  
# Load training samples
train_samples = [HRSample(image = image_path, image_size = image_size) for image_path in train_image_paths]
# Load testing samples
test_samples = []
for image_path in test_image_paths:
    mask_path = image_path.replace('.', '_mask.') # For anomaly samples that need to be evaluated, the Ground Truth Mask also needs to be loaded.
    if os.path.exists(mask_path):
        test_samples.append(HRSample(image = image_path, mask = mask_path, image_size = image_size))
    else:
        test_samples.append(HRSample(image = image_path, image_size = image_size))
```
We use the `MultiResolutionHRImageSpliter` class to split high-resolution images into smaller patches for detection:  
``` 
from hiad.utils.split_and_gather import MultiResolutionHRImageSpliter
indexes = MultiResolutionHRImageSpliter(image_size=image_size, patch_size=patch_size)
``` 
`MultiResolutionHRImageSpliter` returns a set of indexes based on the image size and the patch size. These indexes indicate how HiAD will split the image. We can print these indexes as follows:  
``` 
for index in indexes:
    print(index)
# Output:
{"main_index": {"x": 0, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 512, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 1024, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 1536, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 0, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 512, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 1024, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 1536, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 0, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 512, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 1024, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 1536, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 0, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 512, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 1024, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": null}
{"main_index": {"x": 1536, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": null}
``` 
The original high-resolution image is divided into `(2048 × 2048)/(512 × 512) = 16` patches. Each index records the starting coordinates (x, y) and the size (w, h) of a patch.

HiAD transforms the high-resolution detection task into multiple low-resolution detection tasks. 
A key problem is how many anomaly detection models are needed to perform these low-resolution tasks. 
In HiAD, we use `TaskGenerator` classes to establish a mapping between image patches and detection models. 
In this example, we use the simple `A2OTaskGenerator`, which assigns a single model to detect all image patches:
  
```
from hiad.task import A2OTaskGenerator  

tasks = A2OTaskGenerator().create_tasks(train_samples, indexes)
```
`A2OTaskGenerator` divides all indexes into `tasks` (each task corresponds to one detection model). We can use `PrintTasks` to view the task assignments:
```
from hiad.utils.base import PrintTasks
PrintTasks(tasks)
# Output:
Row: 4, Column: 4, Patch Number: 16, Task Number: 1
| task_0 | task_0 | task_0 | task_0 | 
| task_0 | task_0 | task_0 | task_0 | 
| task_0 | task_0 | task_0 | task_0 | 
| task_0 | task_0 | task_0 | task_0 | 
```
In this example, `A2OTaskGenerator` uses a single model to detect all image patches. As a result, only one task, `task_0`, is created, and it covers all the patches.  

Model training is performed using the `HRTrainer` class:  
```
from hiad.trainer import HRTrainer

trainer = HRTrainer(batch_size=batch_size,
                    tasks=tasks,
                    detector_class=detector_class,
                    config=config,
                    checkpoint_root=checkpoints_root,
                    log_root=log_root,
                    vis_root=vis_root)
trainer.train(train_samples = train_samples, gpu_ids = gpus)
```
The console will output:

```
Start training, devices: [0]
Tasks config is saved as: saved_models/tasks.json
Task: task_0, Indexes: ['{"main_index": {"x": 0, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 512, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1024, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1536, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 0, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 512, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1024, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1536, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 0, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 512, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1024, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1536, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 0, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 512, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1024, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1536, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": null}']
The training progress can be monitored in: logs.
```
HiAD is a highly parallelized framework that distributes multiple models across multiple GPUs for parallel training. 
(In this example, since there is only one model, only a single process will be created.) 
Training progress is not printed to the console but is instead saved to the corresponding log files. 
You can monitor training progress by checking the log files (in this example, they are located in the `logs` directory). 
After training, the model checkpoints will be saved in the `saved_models` directory.

Use the following command for inference and evaluation:
```
from hiad.utils import metrics_gpu # Use GPU to compute evaluation metrics
# from hiad.utils import metrics # Use CPU to compute evaluation metrics
from functools import partial

evaluators = [
     metrics_gpu.compute_imagewise_metrics_gpu,  # Compute image-level metrics
     partial(metrics_gpu.compute_pixelwise_metrics_gpu_resize, resize=512), # Compute pixel-level metrics
 ] 

trainer.inference(test_samples = test_samples,
                    evaluators = evaluators, # Define evaluation method
                    gpu_ids = gpus)
                    
# Output:
|  clsname  |  image_auroc  |  pixel_auroc  |  pixel_ap  |  pixel_f1  |
|:---------:|:-------------:|:-------------:|:----------:|:----------:|
|  unknown  |       1       |   0.991033    |  0.519122  |  0.543303  |
|   mean    |       1       |   0.991033    |  0.519122  |  0.543303  |
```
Visualization results will be saved in the `vis` directory:  
<div align=center><img width="800" src="../assets/vis1.png"/></div>

  
By setting `return_results_only = True` in `trainer.inference`, you can return only the prediction results:  
  
```
image_scores, anomaly_maps = trainer.inference(test_samples=test_samples, return_results_only=True, gpu_ids=gpus)
print(image_scores)
print(len(anomaly_maps), anomaly_maps[0].shape)
Output:
[3.14193442 3.130907   3.21337439 3.28028991 3.56119391 5.3614585
 5.05922744 6.04014021 5.30184198 5.0269954 ]
10 (2048, 2048)
```
If you need online inference (client-server mode), you can refer to this [tutorial](online_inference.md).  
  
The complete code for this example can be found at [quick_start.py](quick_start.py).  
  
**If you'd like to explore more features of HiAD, please continue reading the [Advanced Settings](advanced.md)**.












