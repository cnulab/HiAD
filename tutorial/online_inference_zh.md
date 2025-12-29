# 在线推理
  
本教程将演示HiAD的在线推理和部署流程。  
  
以[快速开始教程](tutorial/quick_start_zh.md)中的示例为例，训练完成后，模型的checkpoint保存在`saved_models`文件夹下：
```
|--saved_models                      
   |--task_0_weight.pkl
   |--tasks.json  
```
  
HiAD根据checkpoint文件和检测器配置创建`HRInferencer`对象：
```
from hiad.inferencer import HRInferencer
from hiad.detectors import HRPatchCore

if __name__ == '__main__':
    detector_class = HRPatchCore
    checkpoint_root = 'saved_models'
    gpus = [0] 
    patch_size = 512
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
    }

    inferencer = HRInferencer(
        detector_class,
        config,
        checkpoint_root = checkpoint_root,
        gpu_ids = gpus,
        models_per_gpu = 1,      #每个GPU中加载的检测器数量，默认值为-1表示将所有模型均摊到各GPU中。
    )

    inferencer.client_inference(ip='127.0.0.1', port=1473)
```
HiAD将在`127.0.0.1:1473`创建在线检测服务：
```
Loading checkpoints...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.97s/it]
HiAD Service Listening...
```
  
在客户端运行以下代码执行在线推理：
```
from hiad.inferencer.client import client_detection

if __name__ == '__main__':

    image = ['data/test000.jpg']
    result = client_detection(image, image_size=2048, ip='127.0.0.1', port = '1473')
    
    print(result["image_scores"])
    print(result['anomaly_maps'].shape)
    
    # output: 
    [3.30944631] 
    (1, 2048, 2048)
```
`client_detection`函数还支持传入`numpy.ndarray`对象：
```
from hiad.inferencer.client import client_detection
from PIL import Image
import numpy as np

if __name__ == '__main__':

    image = Image.open('data/test000.jpg').resize((2048, 2048))
    image = np.array(image)  # (2048, 2048, 3)
    result = client_detection([image], ip='127.0.0.1', port = '1473')
    
    print(result["image_scores"])
    print(result['anomaly_maps'].shape)
    
    # output: 
    [3.30944631] 
    (1, 2048, 2048)
```
