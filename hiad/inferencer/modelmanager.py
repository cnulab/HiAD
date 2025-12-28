import os
import copy
import torch


class ModelManager:

    def __init__(self,
                 tasks,
                 detector_class,
                 config,
                 checkpoint_root: str,
                 gpu_id: int,
                 models_per_gpu: int):

        self.tasks = tasks
        self.detector_class = detector_class
        self.config = config
        self.gpu_device = torch.device("cuda:{}".format(gpu_id))
        self.cpu_device = torch.device("cpu")
        self.models_per_gpu = models_per_gpu

        self.models = []

        for task in self.tasks:
            task_name = task['name']

            if task_name == 'thumbnail':
                thumbnail_config = copy.deepcopy(config.thumbnail)
                thumbnail_config.patch_size = thumbnail_config.pop('thumbnail_size')
                detector = detector_class(**thumbnail_config,  device=self.gpu_device, logger=None, seed=0)
            else:
                detector = detector_class(**copy.deepcopy(config.patch), device=self.gpu_device, logger=None, seed=0)

            checkpoint_path = os.path.join(checkpoint_root, f'{task_name}_weight.pkl')
            detector.load_checkpoint(checkpoint_path)
            detector.to_device(self.cpu_device)

            self.models.append({
                "name": task_name,
                "detector": detector,
                "gpu": False,
            })

        load_task_names = self.get_device_task_names(False)
        if len(load_task_names) > self.models_per_gpu:
            load_task_names = load_task_names[:self.models_per_gpu]

        for load_task_name in load_task_names:
            self.change_model_device(load_task_name, target_device='gpu')

    def get_detector(self, task_name, must_in_gpu = True):
        for model in self.models:
            if model['name'] == task_name:
                if not must_in_gpu or task_name in self.get_device_task_names(gpu=True):
                    return model['detector']
                else:
                    self.change_model_device(self.get_device_task_names(gpu=True)[0], target_device='cpu')
                    self.change_model_device(task_name, target_device='gpu')
                    return model['detector']

    def get_device_task_names(self, gpu: bool):
        return [model["name"] for model in self.models if model["gpu"] == gpu]

    def change_model_device(self, task_name, target_device: str = 'gpu'):
        assert target_device in ['gpu', 'cpu'], "target_device must in [gpu, cpu]"
        for model in self.models:
            if model['name'] == task_name:
                if target_device == 'gpu':
                    model["detector"].to_device(self.gpu_device)
                    model['gpu'] = True
                else:
                    model["detector"].to_device(self.cpu_device)
                    model['gpu'] = False
                return

