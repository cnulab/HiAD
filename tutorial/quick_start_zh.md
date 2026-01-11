# 快速开始
  
本教程通过一个简单的例子使您能够快速了解HiAD的工作原理，运行它只需要一块8GB的GPU和10分钟时间。
### 数据集
首先，我们准备了一个用于演示的微型数据集，它由MVTec-AD数据集中的leather类通过超分辨率和缺陷合成技术构建，图像分辨率为`2048 × 2048`。我们使用这个微型数据集来模拟一个高分辨率异常检测任务，数据集可以通过以下链接下载：  
[data.zip (13MB)](data.zip)  
  

### 安装HiAD
使用以下命令安装HiAD：
```
$ pip install hiad[cuda11] # 适用于 Linux 和 cuda11 
$ pip install hiad[cuda12] # 适用于 Linux 和 cuda12
$ pip install hiad[cuda]   # 适用于 Linux 和 其他cuda版本
$ pip install hiad         # 适用于 Windows
```  
<sub><em>由于依赖项`faiss-gpu`暂不支持Windows系统，因此HiAD的部分功能可能在Windows系统下受限。</em></sub>

### 训练与推理
HiAD可以将现有的异常检测方法扩展到任意高分辨率的场景，`它将高分辨率图像切割成小的图像块分别进行检测，最后汇总结果`。
  
首先定义一些基础变量：

```
import os 

image_size = 2048  # 检测分辨率
patch_size = 512   # 划分的图像块大小
batch_size = 2     # Batch Size
gpus = [0]         # 使用的GPU id，使用多个GPU会显著提升训练和推理速度

data_root = 'data'    # 数据集data.zip解压目录
log_root = 'logs'     # 日志文件保存路径
checkpoints_root = 'saved_models' # 模型检查点保存路径
vis_root = 'vis'      # 可视化结果保存路径

os.makedirs(log_root, exist_ok=True)
os.makedirs(checkpoints_root, exist_ok=True)
os.makedirs(vis_root, exist_ok=True)
```
  
在这个例子中，我们使用HiAD中集成的PatchCore算法进行检测：
```
from hiad.detectors import HRPatchCore  # 导入算法模板类

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
  } # 设置PatchCore中的超参数
```
  
加载用于训练和测试的样本：
```
train_image_paths = [os.path.join(data_root, file) for file in os.listdir(data_root) if file.startswith('train')]
print(train_image_paths)
test_image_paths = [os.path.join(data_root, file) for file in os.listdir(data_root) if file.startswith('test') and file.find('mask')==-1]
print(test_image_paths)
```
数据集中包含10张训练图像和10张测试图像（5张正常样本+5张异常样本）  
``` 
['data/train000.jpg', 'data/train003.jpg', 'data/train004.jpg', 'data/train005.jpg', 'data/train006.jpg', 'data/train008.jpg', 'data/train009.jpg', 'data/train001.jpg', 'data/train002.jpg', 'data/train007.jpg']
['data/test004.jpg', 'data/test000.jpg', 'data/test001.jpg', 'data/test002.jpg', 'data/test003.jpg', 'data/test005.jpg', 'data/test008.jpg', 'data/test009.jpg', 'data/test006.jpg', 'data/test007.jpg']
``` 
HiAD将高分辨率图像封装为`HRSample`类，HRSample类通过图像路径进行初始化： 
```     
from hiad.utils.split_and_gather import HRSample  
# 加载训练样本
train_samples = [HRSample(image = image_path, image_size = image_size) for image_path in train_image_paths]
# 加载测试样本
test_samples = []
for image_path in test_image_paths:
    mask_path = image_path.replace('.', '_mask.') # 对于需要评估的异常样本，还需要读取Ground Truth Mask
    if os.path.exists(mask_path):
        test_samples.append(HRSample(image = image_path, mask = mask_path, image_size = image_size))
    else:
        test_samples.append(HRSample(image = image_path, image_size = image_size))
``` 
我们使用`MultiResolutionHRImageSpliter`类将高分辨率图像切割为小的图像块进行检测：
``` 
from hiad.utils.split_and_gather import MultiResolutionHRImageSpliter
indexes = MultiResolutionHRImageSpliter(image_size=image_size, patch_size=patch_size)
``` 
`MultiResolutionHRImageSpliter`根据图像大小和划分的图像块大小返回一组索引，这些索引将指示HiAD如何对图像进行切割，我们可以打印这些索引：
``` 
for index in indexes:
    print(index)
# 输出：
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
原始高分辨率图像被切割为`(2048 × 2048)/(512 × 512)=16`块，每个索引记录了图像块的起点坐标(x, y)和图像块大小(w, h)。
  
HiAD将高分辨率检测任务转化为多个低分辨率检测任务，一个核心问题在于需要训练多少个异常检测模型来完成这些低分辨率检测任务。在HiAD中，我们使用`TaskGenerator`类来建立图像块和检测模型间的映射关系，在本例中，我们使用简单的`A2OTaskGenerator`，它使用一个模型来检测所有的图像块：
```
from hiad.task import A2OTaskGenerator  

tasks = A2OTaskGenerator().create_tasks(train_samples, indexes)
```
`A2OTaskGenerator`将所有索引划分为`tasks`(每个task对应一个检测模型)，我们可以使用`PrintTasks`查看任务分配情况：
```
from hiad.utils.base import PrintTasks  
PrintTasks(tasks)
# 输出：
Row: 4, Column: 4, Patch Number: 16, Task Number: 1
| task_0 | task_0 | task_0 | task_0 | 
| task_0 | task_0 | task_0 | task_0 | 
| task_0 | task_0 | task_0 | task_0 | 
| task_0 | task_0 | task_0 | task_0 | 
```
在这个示例中，A2OTaskGenerator使用一个模型来检查所有图像块。因此，只创建了一个`task_0`，它涵盖了全部图像块。

模型的训练由`HRTrainer`类完成:
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
此时控制台输出：

```
Start training, devices: [0]
Tasks config is saved as: saved_models/tasks.json
Task: task_0, Indexes: ['{"main_index": {"x": 0, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 512, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1024, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1536, "y": 0, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 0, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 512, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1024, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1536, "y": 512, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 0, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 512, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1024, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1536, "y": 1024, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 0, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 512, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1024, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": null}', '{"main_index": {"x": 1536, "y": 1536, "width": 512, "height": 512}, "low_resolution_indexes": null}']
The training progress can be monitored in: logs.
```  
HiAD是一个高度并行化的框架，它将多个需要训练的模型分配到多个GPU上并行训练（在本例中，由于只有一个模型，只会创建一个进程）。它不会将训练进度输出到控制台，而是将其保存到相应的日志文件中。可以通过查看日志文件来监控训练进度（在本例中日志文件位于logs目录）。训练完成后，模型的检查点将保存到`saved_models`目录中。
  
使用以下命令进行推理和评估：
```
from hiad.utils import metrics_gpu #使用GPU计算评价指标
# from hiad.utils import metrics #使用CPU计算评价指标
from functools import partial

evaluators = [
     metrics_gpu.compute_imagewise_metrics_gpu,  # 计算图像级指标
     partial(metrics_gpu.compute_pixelwise_metrics_gpu_resize, resize=512), # 计算像素级指标
 ] 

trainer.inference(test_samples = test_samples,
                    evaluators = evaluators, # 定义评估方法
                    gpu_ids = gpus)
                    
# 输出：
|  clsname  |  image_auroc  |  pixel_auroc  |  pixel_ap  |  pixel_f1  |
|:---------:|:-------------:|:-------------:|:----------:|:----------:|
|  unknown  |       1       |   0.991033    |  0.519122  |  0.543303  |
|   mean    |       1       |   0.991033    |  0.519122  |  0.543303  |
```
可视化结果将保存在 `vis`目录下：
<div align=center><img width="800" src="..\assets\vis1.png"/></div>

   
通过设置`trainer.inference`中的`return_results_only = True`可以仅返回预测结果：
```
image_scores, anomaly_maps = trainer.inference(test_samples=test_samples, return_results_only=True, gpu_ids=gpus)
print(image_scores)
print(len(anomaly_maps), anomaly_maps[0].shape)
输出：
[3.14193442 3.130907   3.21337439 3.28028991 3.56119391 5.3614585
 5.05922744 6.04014021 5.30184198 5.0269954 ]
10 (2048, 2048)
```
如果需要在线推理（Client-Server），请参考[此教程](online_inference_zh.md)。  
  
本示例的完整的代码见 [quick_start.py](quick_start.py)  
  
**如果您想进一步了解HiAD的更多功能，欢迎继续阅读HiAD的[高级设置](advanced_zh.md)**。







