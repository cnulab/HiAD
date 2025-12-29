# Online Inference
  
This tutorial will demonstrate the online inference and deployment process for HiAD.
  
Using the example from the [Quick Start Tutorial](quick_start.md), after training is complete, the model checkpoint is saved in the `saved_models` folder:
```
|--saved_models                      
   |--task_0_weight.pkl
   |--tasks.json  
```
  
HiAD creates an `HRInferencer` object based on the checkpoint files and detector configuration:
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
        models_per_gpu = 1,      #The number of detectors loaded per GPU. The default value is -1, indicating that all models are evenly distributed across the GPUs.
    )

    inferencer.client_inference(ip='127.0.0.1', port=1473)
```
HiAD will create an online detection service at `127.0.0.1:1473`:
```
Loading checkpoints...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.97s/it]
HiAD Service Listening...
```

Run the following code on the Client side to perform inference:
```
from hiad.inferencer.client import client_detection

if __name__ == '__main__':

    images = ['data/test000.jpg']
    result = client_detection(images, image_size=2048, ip='127.0.0.1', port = '1473')
    
    print(result["image_scores"])
    print(result['anomaly_maps'].shape)
    
    # output: 
    [3.30944631] 
    (1, 2048, 2048)
```
The `client_detection` function also supports passing `numpy.ndarray` objects:
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
