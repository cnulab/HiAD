import os
from functools import partial
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from hiad.utils.base import PrintTasks
from hiad.utils.split_and_gather import HRSample, MultiResolutionHRImageSpliter
from hiad.utils import metrics_gpu

from hiad.task import A2OTaskGenerator
from hiad.trainer import HRTrainer
from hiad.detectors import HRPatchCore

if __name__ == '__main__':

    image_size = 2048
    patch_size = 512
    batch_size = 2
    gpus = [0]
    detector_class = HRPatchCore

    log_root = 'logs'
    checkpoints_root = 'saved_models'
    vis_root = 'vis'
    data_root = 'tutorial/data'

    os.makedirs(log_root, exist_ok=True)
    os.makedirs(checkpoints_root, exist_ok=True)
    os.makedirs(vis_root, exist_ok=True)

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

    train_image_paths = [os.path.join(data_root, file) for file in os.listdir(data_root) if file.startswith('train')]
    print(train_image_paths)
    test_image_paths = [os.path.join(data_root, file) for file in os.listdir(data_root) if file.startswith('test') and file.find('mask')==-1]
    print(test_image_paths)

    train_samples = [HRSample(image = image_path, image_size = image_size) for image_path in train_image_paths]
    test_samples = []
    for image_path in test_image_paths:
        mask_path = image_path.replace('.', '_mask.')
        if os.path.exists(mask_path):
            test_samples.append(HRSample(image = image_path, mask=mask_path,image_size = image_size))
        else:
            test_samples.append(HRSample(image = image_path, image_size = image_size))

    indexes = MultiResolutionHRImageSpliter(image_size=image_size, patch_size=patch_size)
    for index in indexes:
        print(index)

    tasks = A2OTaskGenerator().create_tasks(train_samples, indexes)
    PrintTasks(tasks)

    trainer = HRTrainer(batch_size=batch_size,
                        tasks=tasks,
                        checkpoint_root=checkpoints_root,
                        log_root=log_root,
                        detector_class=detector_class,
                        vis_root=vis_root,
                        config=config)

    trainer.train(train_samples = train_samples, gpu_ids = gpus)

    evaluators = [
        metrics_gpu.compute_imagewise_metrics_gpu,
        partial(metrics_gpu.compute_pixelwise_metrics_gpu_resize, resize=512),
    ]

    trainer.inference(test_samples = test_samples,
                       evaluators =evaluators,
                       gpu_ids = gpus)

    image_scores, anomaly_maps = trainer.inference(test_samples=test_samples, return_results_only=True, gpu_ids=gpus)

    print(image_scores)
    print(len(anomaly_maps), anomaly_maps[0].shape)