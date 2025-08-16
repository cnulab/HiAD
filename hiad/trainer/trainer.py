import logging
import os
import cv2
import numpy as np
from typing import List, Dict, Union
import copy
import torch
from tqdm import tqdm
import torch.multiprocessing as mp
from easydict import EasyDict

from hiad.utils.base import create_logger
from hiad.utils.base import split_list, save_tasks_config, load_tasks
from hiad.utils.split_and_gather import NPImageGather, HRSample
from hiad.utils.device import train_in_device, test_in_device, compute_metrics_in_device
from hiad.utils.base import create_share_array, Report
from hiad.task.utils import update_paths_and_indexes

class HRTrainer:
    def __init__(self,
                 detector_class,
                 config,
                 batch_size: int,
                 checkpoint_root: str,
                 log_root: str,
                 tasks: List[Dict] = None,
                 vis_root: str = None,
                 seed: int = 0,
                 fusion_weights: List = None,
                 early_stop_epochs = -1):
        r"""
           High-resolution trainer

           Args:
               detector_class (class): The provided detector class template, used to create detector instances across multiple processes.
                    Supported detector types are listed in `detectors.__init__`.
               config (dict): Detector configuration parameters.
               batch_size (size): Batch Size
               checkpoint_root (str): Directory for saving model checkpoints
               log_root (str): Directory for saving Logs
               vis_root (str): Directory for saving visualization results. If None, visualization of results will be ignored.
               tasks (List[Dict]): The provided task list are created by the TaskGenerator. Supported TaskGenerators include:
                                   O2OTaskGenerator, A2OTaskGenerator, NeighborhoodTaskGenerator, SpatialClusteringTaskGenerator,
                                   ThresholdClusteringTaskGenerator, and RetrieveTaskGenerator.
                                   If None, it will be loaded from the checkpoint.
               seed (int): random seed.
               fusion_weights (list): Weights used for feature fusion. If None, a set of equal weights will be used.
               early_stop_epochs (int): Controls early stopping for detector: if a detector shows no performance improvement on the validation set for N epochs, training will be stopped.
                                        If -1, Early stopping will be disabled.
        """

        self.detector_class = detector_class
        self.batch_size = batch_size
        self.checkpoint_root = checkpoint_root
        self.log_root = log_root
        self.vis_root = vis_root
        self.seed = seed
        self.tasks = tasks
        self.early_stop_epochs = early_stop_epochs
        self.fusion_weights = fusion_weights

        if type(config) == dict:
            config = EasyDict(config)
        self.config = config
        self.use_thumbnail = 'thumbnail' in self.config
        mp.set_start_method('spawn', force=True)

    def train(self,
              train_samples: List[HRSample],
              gpu_ids: List[int],
              main_logger = None,
              val_config: Dict = None):
        r"""
         Args:
             train_samples (List[HRSample]): List of provided high-resolution training samples.
             gpu_ids (List[int]): List of GPUs id.
             main_logger (logging.Logger): Logger object to use. If None, a new one will be created.
             val_config (dict): This configuration is used to split the validation set, used for checkpoint selection and normalization.
                For example:
                    val_config = {
                            'anomaly_syn_fn': RandomeBoxSynthesizer(p=0.5), # Defines the anomaly synthesis method used to create anomaly samples.
                            'val_sample_ratio': 0.2, # Number of samples from `train_samples` to be used as the validation set. (0.2 means 20%)
                            'max_patch_number': 200, # Maximum number of validation patches per task. The default value is -1, which means no limit.
                            'evaluators': evaluators # Validation set evaluation methods.
                        }
                If None, validation will be disabled.
        """
        assert self.tasks is not None, 'tasks is None'

        if main_logger is None:
            main_logger = create_logger("main", os.path.join(self.log_root, 'main.log'), print_console=True)

        main_logger.info(f'Start training, devices: {gpu_ids}')

        if self.use_thumbnail:
            thumbnail_task = {'name': 'thumbnail', 'thumbnail_size': self.config.thumbnail.thumbnail_size}
            self.tasks.append(thumbnail_task)
            main_logger.info('Add thumbnail task: {}'.format(thumbnail_task))

        tasks_save_path = os.path.join(self.checkpoint_root, 'tasks.json')
        save_tasks_config(self.tasks, tasks_save_path)
        main_logger.info(f'Tasks config is saved as: {tasks_save_path}')

        for index, task in enumerate(self.tasks):
            if task['name'].startswith('retrieve'):
                main_logger.info(f"[{index+1}/{len(self.tasks)}] Task: {task['name']}, patch_number: {len(task['paths_and_indexes'])}")
            elif task['name'] == 'thumbnail':
                main_logger.info(f"[{index+1}/{len(self.tasks)}] Task: {task['name']}, thumbnail_size: {task['thumbnail_size']}")
            else:
                main_logger.info(f"[{index+1}/{len(self.tasks)}] Task: {task['name']}, Indexes: { [ str(index) for index in task['indexes']]}")

        main_logger.info(f'The training progress can be monitored in: {self.log_root}.')

        num_device = len(gpu_ids)
        tasks_in_device = split_list(self.tasks, num_device)

        results = []
        process_pool = mp.Pool(processes = num_device)
        for gpu_id, task in zip(gpu_ids, tasks_in_device):
            results.append(process_pool.apply_async(train_in_device, args=(gpu_id, self.detector_class, self.config,
                            copy.deepcopy(train_samples), task, self.batch_size,
                            self.checkpoint_root, self.log_root, self.seed, copy.deepcopy(val_config), self.fusion_weights, self.early_stop_epochs)))

        process_pool.close()
        process_pool.join()
        for result in results:
            result = result.get()
            if result is not None:
                main_logger.info(result)

        main_logger.info(f'End training')
        main_logger.info(f'Checkpoints are saved as: {self.checkpoint_root}')


    def inference(self,
                  test_samples: List[HRSample],
                  gpu_ids: List[int],
                  return_results_only = False,
                  evaluators: List = None,
                  main_logger = None,
                  vis_size: Union[int, List] = 1024):

        r"""
              Args:
                  test_samples (List[HRSample]): List of provided high-resolution testing samples.
                  gpu_ids (List[int]): List of GPUs id.
                  main_logger (logging.Logger): Logger object to use. If None, a new one will be created.
                  return_results_only (bool): Return results only. Default: False
                  evaluators: evaluation methods. Setting this to None is only allowed when `return_results_only` is set to True.
                  vis_size: Resolution of the saved visualization images; Default: 1024×1024.
        """

        if os.path.exists(os.path.join(self.checkpoint_root, 'tasks.json')):
            self.tasks = load_tasks(os.path.join(self.checkpoint_root, 'tasks.json'))
            if not self.use_thumbnail:
                self.tasks = [task for task in self.tasks if task['name'] != 'thumbnail']

        if main_logger is None:
            main_logger = create_logger("main", os.path.join(self.log_root, 'main.log'), print_console=True)

        main_logger.info(f'Start inference, devices: {gpu_ids}')
        main_logger.info(f'The inference progress can be monitored in : {self.log_root}.')

        num_device = len(gpu_ids)
        retrieve_tasks = [task for task in self.tasks if task['name'].startswith('retrieve')]

        if len(retrieve_tasks) != 0:
            main_logger.info(f'Start Retrieve')
            retrieve_tasks = update_paths_and_indexes(
                                     copy.deepcopy(retrieve_tasks),
                                     test_samples,
                                     self.detector_class,
                                     self.config.patch,
                                     self.batch_size,
                                     gpu_ids)

            tasks = [task for task in retrieve_tasks if len(task['paths_and_indexes']) != 0] + [task for task in self.tasks if not task['name'].startswith('retrieve')]
            main_logger.info(f'Retrieve End')
        else:
            tasks = self.tasks

        process_pool = mp.Pool(processes=num_device)
        tasks_in_device = split_list(tasks, num_device)

        device_results = []
        for gpu_id, task in zip(gpu_ids, tasks_in_device):
            device_result = process_pool.apply_async(test_in_device, (gpu_id, self.detector_class, self.config,
                                                            copy.deepcopy(test_samples), task, self.batch_size,
                                                            self.checkpoint_root, self.log_root, self.seed))
            device_results.append(device_result)

        process_pool.close()
        process_pool.join()

        prediction_masks = {sample.image.image_path: {"patch_list": [], "index_list": []} for sample in test_samples}

        for device_result in device_results:
            device_result = device_result.get()
            for key in device_result:
                prediction_masks[key]['patch_list'].extend(device_result[key]['patch_list'])
                prediction_masks[key]['index_list'].extend(device_result[key]['index_list'])
                if 'thumbnail' in device_result[key]:
                    prediction_masks[key]['thumbnail'] = device_result[key]['thumbnail']

        patch_prediction_masks = {}
        thumbnail_prediction_masks = {}

        for key in tqdm(prediction_masks, desc='Gather Segmentations'):
            segmentation = NPImageGather(patch_list = prediction_masks[key]['patch_list'],
                                         index_list = prediction_masks[key]['index_list'])
            patch_prediction_masks[key] = segmentation
            if 'thumbnail' in prediction_masks[key]:
                thumbnail_prediction_masks[key] = prediction_masks[key]['thumbnail']

        patch_prediction_masks = np.stack([patch_prediction_masks[sample.image.image_path] for sample in test_samples])

        if thumbnail_prediction_masks != {}:
            thumbnail_prediction_masks = np.stack(
                [thumbnail_prediction_masks[sample.image.image_path] for sample in test_samples])
            assert thumbnail_prediction_masks.shape[0] == patch_prediction_masks.shape[0]
        else:
            thumbnail_prediction_masks = None

        prediction_masks = self.detector_class.post_processing(patch_prediction_masks, thumbnail_prediction_masks,
                                                               is_normalize= not return_results_only,
                                                               gaussian_filter = True)
        prediction_scores = self.detector_class.get_image_score(prediction_masks, batch_size = self.batch_size)

        if return_results_only:
            return prediction_scores, [prediction_masks[i] for i in range(prediction_masks.shape[0])]

        assert evaluators is not None, "No evaluation function provided."

        gt_masks = []
        gt_labels = []

        for sample in tqdm(test_samples, desc='Loading GT Masks'):
            if sample.mask is not None:
                sample.open()
                mask = copy.deepcopy(sample.mask.image)
                mask[mask != 0] = 1
                sample.close()
            else:
                mask = np.zeros(prediction_masks.shape[-2:]).astype(float)

            if sample.label is not None:
                gt_labels.append(sample.label)
            else:
                gt_labels.append(np.max(mask).item())

            gt_masks.append(mask)

        gt_masks = np.stack(gt_masks)

        clsname_list = [sample.clsname if sample.clsname is not None else 'unknown' for sample in test_samples]
        all_clsnames = list(set(clsname_list))

        torch.cuda.empty_cache()
        main_logger.info(f'Computing Metrics')

        device_results = []

        process_pool = mp.Pool(processes=num_device)
        clsnames_in_device = split_list(all_clsnames, num_device)

        prediction_masks = create_share_array(prediction_masks)
        gt_masks = create_share_array(gt_masks)

        for gpu_id, clsnames in zip(gpu_ids, clsnames_in_device):
            device_result = process_pool.apply_async(compute_metrics_in_device, (gpu_id, clsnames, evaluators,
                                                                                 prediction_masks, gt_masks,
                                                                                 gt_labels, prediction_scores,
                                                                                 clsname_list))
            device_results.append(device_result)

        process_pool.close()
        process_pool.join()

        scores = []
        for device_result in device_results:
            cls_score = device_result.get()
            scores.extend(cls_score)

        scores.sort(key = lambda a: a['clsname'])

        all_metrics = [item for item in scores[0] if item.find('threshold') == -1 and item.find('clsname') == -1]

        record = Report(["clsname"] + all_metrics)

        for score in scores:
            clsvalues = [
                score[metric] for metric in all_metrics
            ]
            record.add_one_record([score['clsname']] + clsvalues)

        mean_metrics = []
        for metric in all_metrics:
            mean_metrics.append(np.mean([score[metric] for score in scores]))

        record.add_one_record(['mean'] + mean_metrics)

        main_logger.info(f"\n{record}")

        if self.vis_root is not None:

            vis_size = (vis_size, vis_size) if isinstance(vis_size, int) else vis_size

            from skimage.segmentation import mark_boundaries
            from PIL import Image
            from pytorch_grad_cam.utils.image import show_cam_on_image

            main_logger.info(f'Starting save Vis ...')

            if 'seg_threshold' in scores[0]:
                seg_thresholds = {score['clsname']: score['seg_threshold'] for score in scores}
            else:
                seg_thresholds = None

            prediction_masks = prediction_masks.numpy() if isinstance(prediction_masks, torch.Tensor) else prediction_masks
            gt_masks = gt_masks.numpy() if isinstance(gt_masks, torch.Tensor) else gt_masks

            for index, (sample, pred_mask, gt_mask, clsname) in enumerate(tqdm(zip(test_samples, prediction_masks, gt_masks, clsname_list), total=len(test_samples))):
                sample.open()

                image = sample.down_sampling_to_LR(vis_size).image
                pred_mask = cv2.resize(pred_mask, vis_size, interpolation=cv2.INTER_NEAREST)
                gt_mask = cv2.resize(gt_mask,  vis_size, interpolation=cv2.INTER_NEAREST)

                heat = show_cam_on_image(image / 255, pred_mask, use_rgb=True)

                if seg_thresholds is not None:
                    pred_mask[pred_mask >= seg_thresholds[clsname]] = 1
                    pred_mask[pred_mask < seg_thresholds[clsname]] = 0
                    image_with_pred = mark_boundaries(image / 255, pred_mask, color=(1, 0, 0), mode='inner')
                    image_with_mask = mark_boundaries(image / 255, gt_mask, color=(1, 0, 0), mode='inner')
                    save_image = np.concatenate([image_with_pred * 255, heat, image_with_mask * 255], axis=1)
                else:
                    image_with_mask = mark_boundaries(image / 255, gt_mask, color=(1, 0, 0), mode='inner')
                    save_image = np.concatenate([image, heat, image_with_mask * 255], axis=1)

                _, image_name = os.path.split(sample.image.image_path)
                Image.fromarray(save_image.astype(np.uint8)).save(os.path.join(self.vis_root, "{}{}_{}".format(clsname, index, image_name)))
                sample.close()

        main_logger.info(f'End Inference')
