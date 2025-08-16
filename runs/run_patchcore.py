import copy
import json
import shutil
import yaml
import argparse
import os
from functools import partial
from easydict import EasyDict
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from hiad.utils.base import read_meta_file, create_logger, PrintTasks
from hiad.utils.split_and_gather import HRSample, MultiResolutionHRImageSpliter
from hiad.utils import metrics_gpu
from hiad.task import A2OTaskGenerator, O2OTaskGenerator, NeighborhoodTaskGenerator, SpatialClusteringTaskGenerator, RetrieveTaskGenerator
from hiad.trainer import HRTrainer
from hiad.detectors import HRPatchCore



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="High resolution Anomaly Detection")
    parser.add_argument("--data_root", default="data/MVTec-2K", help='dataset path')
    parser.add_argument("--category", default="bottle", help='category to detect')

    parser.add_argument("--image_size", default = 2048, help='Detection resolution: images will be resized to this size for detection')
    parser.add_argument("--patch_size", default = 512, help='Patch size: size of the image patches after cropping')
    parser.add_argument("--stride", default = -1 , help='Stride size: set to -1 to use the same value as the patch size')
    parser.add_argument("--ds_factors", default = [0, 1], type = list, help='Downsampling ratio used for feature fusion')
    parser.add_argument("--fusion_weights", default = [0.5, 0.5], type = list, help='Fusion weights for feature fusion')
    parser.add_argument("--batch_size", default = 8, type = int, help='Batch size')

    parser.add_argument("--use_thumbnail", default = True, type=bool, help='Enable thumbnail detection (low-resolution detection)')
    parser.add_argument("--thumbnail_size", default = 512, help='low-resolution detection resolution')
    parser.add_argument("--thumbnail_config", default = 'configs/patchcore.yaml', type= str, help='Low-res(thumbnail) detection model configuration path')

    parser.add_argument("--detector_number", default = 4, type=int, help='number of detector')
    parser.add_argument("--evaluation_only", default = False, type=bool, help='Run evaluation only')
    parser.add_argument("--early_stop_epochs", default = -1, type = int,
                        help='Controls early stopping for detector: if a detector shows no performance improvement on the validation set for N epochs, training will be stopped. If -1, Early stopping will be disabled.')

    parser.add_argument("--detector_config_path", default = 'configs/patchcore.yaml', type = str, help='High-res detection model configuration path')
    parser.add_argument("--seed", default = 42, type = int)

    parser.add_argument("--checkpoint_root", default ='results/patchcore_checkpoints', type = str, help='Directory for saving model checkpoints')
    parser.add_argument("--log_root", default = 'results/patchcore_logs', type = str, help='Directory for saving logs')
    parser.add_argument("--vis_root", default = 'results/patchcore_vis', type = str, help='Directory for saving visualization results')

    parser.add_argument("--gpus", default="0,1,2,3,4", type=str, help='gpu ids')

    args = parser.parse_args()
    args.gpus = [int(gpu_id) for gpu_id in args.gpus.split(',')]

    image_size = args.image_size
    patch_size = args.patch_size
    stride = args.stride

    category = args.category
    data_root = args.data_root

    detector_number = args.detector_number
    early_stop_epochs = args.early_stop_epochs

    args.checkpoint_root = os.path.join(args.checkpoint_root, args.category)
    args.log_root = os.path.join(args.log_root, args.category)
    args.vis_root = os.path.join(args.vis_root)

    os.makedirs(args.checkpoint_root, exist_ok=True)
    os.makedirs(args.log_root, exist_ok=True)
    os.makedirs(args.vis_root, exist_ok=True)

    with open(os.path.join(args.checkpoint_root, 'args.json'), 'w+') as f:
        json.dump(vars(args), f, indent=4)

    shutil.copy(args.detector_config_path,
                os.path.join(args.checkpoint_root, os.path.basename(args.detector_config_path)))

    if args.use_thumbnail and args.thumbnail_config is not None:
        shutil.copy(args.thumbnail_config,
                    os.path.join(args.checkpoint_root, os.path.basename(args.thumbnail_config)))

    with open(args.detector_config_path) as f:
        patch_detector_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    if args.use_thumbnail:
        if args.thumbnail_config is not None:
            with open(args.thumbnail_config) as f:
                thumbnail_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        else:
            thumbnail_config = copy.deepcopy(patch_detector_config)

        thumbnail_size = args.thumbnail_size if args.thumbnail_size is not None else args.patch_size
        thumbnail_config.thumbnail_size = thumbnail_size

        config = EasyDict(patch=patch_detector_config, thumbnail=thumbnail_config)
    else:
        config = EasyDict(patch=patch_detector_config)

    config.patch.patch_size = args.patch_size

    main_logger = create_logger("main", os.path.join(args.log_root, 'main.log'), print_console=True)
    main_logger.info(args)
    main_logger.info(config)

    train_meta_path = os.path.join(data_root, category, 'train.jsonl')
    test_meta_path = os.path.join(data_root, category, 'test.jsonl')

    train_samples = read_meta_file(train_meta_path)
    test_samples = read_meta_file(test_meta_path)

    train_samples = [HRSample(image = os.path.join(args.data_root, sample['filename']),
                              image_size = image_size,
                              clsname = sample['clsname'],
                              label = sample['label'],
                              label_name = sample['label_name']) for sample in train_samples]

    test_samples = [HRSample( image = os.path.join(args.data_root,sample['filename']),
                              mask = os.path.join(args.data_root, sample['mask']) if 'mask' in sample else None,
                              image_size = image_size,
                              clsname = sample['clsname'],
                              label = sample['label'],
                              label_name = sample['label_name']) for sample in test_samples]

    main_logger.info(f"train dataset len is: {len(train_samples)}")
    main_logger.info(f"test dataset len is: {len(test_samples)}")

    detector_class = HRPatchCore
    main_logger.info(f"Anomaly detector is: {detector_class}")

    if not args.evaluation_only:

        image_indexes = MultiResolutionHRImageSpliter(image_size = image_size, patch_size=patch_size,
                                                      ds_factors = args.ds_factors,
                                                      stride = stride if stride != -1 else None)

        tasks = O2OTaskGenerator().create_tasks(train_samples, image_indexes)
        PrintTasks(tasks)
    else:
        tasks = None

    trainer = HRTrainer(batch_size = args.batch_size,
                        tasks = tasks,
                        checkpoint_root = args.checkpoint_root,
                        log_root = args.log_root,
                        vis_root = args.vis_root,
                        seed = args.seed,
                        detector_class = detector_class,
                        config = config,
                        fusion_weights = args.fusion_weights,
                        early_stop_epochs = early_stop_epochs)

    if not args.evaluation_only:
        trainer.train(
                      train_samples = train_samples,
                      gpu_ids = args.gpus,
                      main_logger=main_logger,
        )

    evaluators = [
        metrics_gpu.compute_imagewise_metrics_gpu,
        partial(metrics_gpu.compute_pixelwise_metrics_gpu_resize, resize = 512),
        partial(metrics_gpu.compute_pro_gpu_resize, resize = 512),
    ]

    trainer.inference(test_samples = test_samples,
                      evaluators = evaluators,
                      main_logger=main_logger,
                      gpu_ids = args.gpus,
                      vis_size = 1024)