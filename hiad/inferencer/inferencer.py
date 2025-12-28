import os
import cv2
import numpy as np
from typing import List, Dict, Union
import copy
import torch
import math
import torch.multiprocessing as mp
from easydict import EasyDict
from tqdm import tqdm
import traceback
import zmq

from hiad.utils.base import split_list, load_tasks
from hiad.utils.split_and_gather import NPImageGather, HRSample, HRImage
from hiad.inferencer.modelmanger import ModelManager
from hiad.inferencer.retrieve import update_paths_and_indexes_at_inferencer


def inference_in_device(test_samples,
                        task_in_device,
                        model_manager):
    device_results = {sample.image.image_path: {"patch_list": [], "index_list": []} for sample in test_samples}
    test_data = {}
    batch_size = len(test_samples)

    retrieve_tasks = [task for task in task_in_device if task['name'].startswith('retrieve')]

    if len(retrieve_tasks) != 0:
        retrieve_tasks = update_paths_and_indexes_at_inferencer(
            copy.deepcopy(retrieve_tasks),
            test_samples,
            model_manager,
            batch_size)

    task_in_device = [task for task in retrieve_tasks if len(task['paths_and_indexes']) != 0] \
                     + [task for task in task_in_device if not task['name'].startswith('retrieve')]

    for i, sample in enumerate(test_samples):

        for task in task_in_device:
            task_name = task['name']

            if task_name.startswith('retrieve'):
                if task_name not in test_data:
                    test_data[task_name] = []

                paths_and_indexes = task['paths_and_indexes']
                sample_map = {}
                for path, index in paths_and_indexes:
                    if path not in sample_map:
                        sample_map[path] = []
                    sample_map[path].append(index)
                if sample.get_image_path() in sample_map:
                    indexes = sample_map[sample.get_image_path()]
                    for index in indexes:
                        data = sample[index.main_index]
                        if index.low_resolution_indexes is not None:
                            for low_resolution_index in index.low_resolution_indexes:
                                data.add_low_resolution_images(low_resolution_index, sample.image)
                        test_data[task_name].append([sample.get_image_path(), index, data])

            elif task_name == 'thumbnail':
                if task_name not in test_data:
                    test_data[task_name] = []
                test_data[task_name].append(sample.down_sampling_to_LR(task['thumbnail_size']))
            else:
                if task_name not in test_data:
                    test_data[task_name] = {}
                indexes = task['indexes']
                for index in indexes:
                    if index not in test_data[task_name]:
                        test_data[task_name][index] = []
                    data = sample[index.main_index]
                    if index.low_resolution_indexes is not None:
                        for low_resolution_index in index.low_resolution_indexes:
                            data.add_low_resolution_images(low_resolution_index, sample.image)
                    test_data[task_name][index].append(data)

    sorted_names = model_manager.get_device_task_names(gpu=True) + model_manager.get_device_task_names(gpu=False)
    task_in_device = {task['name']: task for task in task_in_device}
    task_in_device = [task_in_device[name] for name in sorted_names]

    for task in task_in_device:
        task_name = task['name']

        if task_name.startswith('retrieve'):

            paths = [path for path, index, data in test_data[task_name]]
            indexes = [index for path, index, data in test_data[task_name]]
            lr_sampls = [data for path, index, data in test_data[task_name]]

            detector = model_manager.get_detector(task_name)
            test_dataset = detector.create_dataset(lr_sampls, training=False, task_name=task_name)
            test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
            )
            segmentation_patchs = detector.inference_step(test_dataloader, task_name)

            assert len(segmentation_patchs) == len(paths) == len(indexes)
            for path, index, segmentation_patch in zip(paths, indexes, segmentation_patchs):
                device_results[path]['patch_list'].append(segmentation_patch)
                device_results[path]['index_list'].append(index.main_index)

        elif task_name == 'thumbnail':
            lr_sampls = test_data[task_name]
            detector = model_manager.get_detector(task_name)
            test_dataset = detector.create_dataset(lr_sampls, training=False, task_name=task_name)

            test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers = 0,
                    pin_memory=True,
            )

            segmentation_patchs = detector.inference_step(test_dataloader, task_name)
            assert len(segmentation_patchs) == len(test_samples)
            for sample, segmentation_patch in zip(test_samples, segmentation_patchs):
                device_results[sample.image.image_path]['thumbnail'] = segmentation_patch

        else:
            indexes = task['indexes']
            for index in indexes:
                lr_sampls = test_data[task_name][index]
                detector = model_manager.get_detector(task_name)

                test_dataset = detector.create_dataset(lr_sampls, training=False, task_name=task_name)
                test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )

                segmentation_patchs = detector.inference_step(test_dataloader, task_name)
                assert len(segmentation_patchs) == len(test_samples)
                for sample, segmentation_patch in zip(test_samples, segmentation_patchs):
                    device_results[sample.image.image_path]['patch_list'].append(segmentation_patch)
                    device_results[sample.image.image_path]['index_list'].append(index.main_index)

    return device_results



class HRInferencer:

    def __init__(self,
                 detector_class,
                 config,
                 checkpoint_root: str,
                 gpu_ids: List[int],
                 models_per_gpu: int = -1):

        r"""
           High-resolution trainer

           Args:
               detector_class (class): The provided detector class template, used to create detector instances across multiple processes.
                    Supported detector types are listed in `detectors.__init__`.
               config (dict): Detector configuration parameters.
               checkpoint_root (str): Directory for saving model checkpoints
               gpu_ids (List[int]): GPU indexes list.
               models_per_gpu (int): Number of models loaded per GPU. -1: load all models.
        """
        self.detector_class = detector_class
        self.checkpoint_root = checkpoint_root
        if type(config) == dict:
            config = EasyDict(config)
        self.config = config
        self.use_thumbnail = 'thumbnail' in self.config
        self.gpu_ids = gpu_ids
        self.models_per_gpu = models_per_gpu
        assert self.models_per_gpu > 0 or self.models_per_gpu == -1

        mp.set_sharing_strategy("file_system")
        mp.set_start_method('spawn', force=True)

        if os.path.exists(os.path.join(self.checkpoint_root, 'tasks.json')):
            self.tasks = load_tasks(os.path.join(self.checkpoint_root, 'tasks.json'))
            if not self.use_thumbnail:
                self.tasks = [task for task in self.tasks if task['name'] != 'thumbnail']

        self.tasks_in_devices = [tasks_in_device for tasks_in_device in split_list(self.tasks, len(self.gpu_ids)) if len(tasks_in_device) != 0]

        if self.models_per_gpu == -1:
            self.models_per_gpu = max([len(tasks_in_device) for tasks_in_device in self.tasks_in_devices])

        self.model_managers = []

        for idx, tasks_in_device in tqdm(enumerate(self.tasks_in_devices), total=len(self.tasks_in_devices), desc = "Loading checkpoints..."):
            gpu_id = self.gpu_ids[idx]
            self.model_managers.append(ModelManager(tasks_in_device, self.detector_class, self.config, self.checkpoint_root, gpu_id, self.models_per_gpu))


    def inference(self, test_samples: List[HRSample]):

        for sample in test_samples:
            assert sample.image.image is not None, "Please ensure all HRsample are opened: sample.open()"

        process_pool = mp.Pool(processes = len(self.tasks_in_devices))
        device_results = []

        for model_manager, task in zip(self.model_managers, self.tasks_in_devices):
            device_result = process_pool.apply_async(inference_in_device, (copy.deepcopy(test_samples), task, model_manager))
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
            segmentation = NPImageGather(patch_list=prediction_masks[key]['patch_list'],
                                         index_list=prediction_masks[key]['index_list'])
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
                                                               is_normalize=False, gaussian_filter=True)
        prediction_scores = self.detector_class.get_image_score(prediction_masks)

        return prediction_scores, [prediction_masks[i] for i in range(prediction_masks.shape[0])]


    def client_inference(self, ip='127.0.0.1', port = '1473'):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://{}:{}".format(ip, port))
        print("HiAD Service Listening...")

        while True:
            try:
                images = socket.recv_pyobj()
                samples = []

                for idx, image in enumerate(images):
                    sample = HRSample(image = HRImage(image_path = str(idx)))
                    sample.image.image = image
                    samples.append(sample)

                image_scores, anomaly_maps = self.inference(samples)

                response = {
                    'status': 'success',
                    'image_scores': image_scores,
                    'anomaly_maps': anomaly_maps
                }

            except Exception as e:

                traceback.print_exc()
                response = {
                    'status': 'error',
                    'msg': str(e),
                    'traceback': traceback.format_exc()
                }

            socket.send_pyobj(response)
