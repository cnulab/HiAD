from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import torch
import tqdm
import numpy
import logging
from sklearn.cluster import DBSCAN, KMeans
import torch.nn.functional as F
import os
from PIL import Image
from easydict import EasyDict

from hiad.task.utils import patch_embedding
from hiad.utils.split_and_gather import HRSample, HRImage
from hiad.utils.split_and_gather import MultiResolutionIndex
from hiad.utils.base import ImageIndexes2NdArray


class BaseTaskGenerator(ABC):
    r"""
       The TaskGenerator divides image patches into different tasks for processing,
         with each task assigned to a detector. This class serves as the base class for all TaskGenerator implementations

       Args:
           deviec (List[int]): GPU ids.
    """

    def __init__(self, device = None):
        self.device = device

    @abstractmethod
    def create_tasks(self,
                        samples: List[HRSample],
                        indexes: List[MultiResolutionIndex],
                        **kwargs) -> List[Dict]:
        r"""
           The core method for task splitting
           Args:
               samples (List[HRSample]): The provided image set, typically the training set.
               indexes (List[MultiResolutionIndex]): indexes set.
           returns:
               Returns the list of created tasks. List(dict)
        """
        raise NotImplementedError


class O2OTaskGenerator(BaseTaskGenerator):
    r"""
        Assign all indexes to a single task.
    """

    def create_tasks(self,
                        samples: List[HRSample],
                        indexes: List[MultiResolutionIndex],
                        **kwargs) -> List[Dict]:

        return [{"name": f'task_{i}', "indexes": [index]} for i, index in enumerate(indexes)]


class A2OTaskGenerator(BaseTaskGenerator):
    r"""
          Assign each index to a separate task.
    """
    def create_tasks(self,
                        samples: List[HRSample],
                        indexes: List[MultiResolutionIndex],
                        **kwargs) -> List[Dict]:
        return [{"name": f'task_0', "indexes": indexes}]


class NeighborhoodTaskGenerator(BaseTaskGenerator):
    r"""
        Assign all indexes within a small spatial neighborhood to a single task.

        Args:
            num_groups_width (int): Number of neighborhoods divided along the image width. if -1, will be set to the number of patches.
            num_groups_height (int): Number of neighborhoods divided along the image height. if -1, will be set to the number of patches.
                if None, will be set to the `num_groups_width`.
            deviec (List[int]): GPU ids.
     """

    def __init__(self, num_groups_width, num_groups_height = None, device=None):
        assert num_groups_width is not None
        self.num_groups_row = num_groups_width
        if num_groups_height is None:
            self.num_groups_column = self.num_groups_row
        else:
            self.num_groups_column = num_groups_height
        assert (self.num_groups_row > 0 or self.num_groups_row == -1) and (self.num_groups_column > 0 or self.num_groups_column == -1)
        super().__init__(device)

    def create_tasks(self,
                        samples: List[HRSample],
                        indexes: List[MultiResolutionIndex],
                        **kwargs) -> List[Dict]:

        indexes_array = ImageIndexes2NdArray(indexes)

        if self.num_groups_row == -1:
            num_row = indexes_array.shape[0]
        else:
            num_row = self.num_groups_row

        if self.num_groups_column == -1:
            num_column = indexes_array.shape[1]
        else:
            num_column = self.num_groups_column

        def split_array(arr, m, n):
            row_splits = numpy.array_split(arr, m, axis=0)
            result = [numpy.array_split(row, n, axis=1) for row in row_splits]
            return [submat.flatten() for row in result for submat in row]

        tasks = split_array(indexes_array,num_column, num_row)
        tasks = [task for task in tasks if len(task) != 0]
        tasks = [{'name': f'task_{index}', 'indexes': task} for index, task in enumerate(tasks)]
        return tasks


class SpatialClusteringTaskGenerator(BaseTaskGenerator):
    r"""
        Cluster spatial patches with high visual similarity.

        Args:
            detector_class (class): The provided detector class template, used to create detector instances across multiple processes.
                    Supported detector types are listed in `detectors.__init__`.
            detector_config: Detector configuration parameters.
            batch_size (int): Feature extraction Batch Size.
            cluster_number (int): Number of clusters.
            feature_resolution (int): The extracted multi-scale features are uniformly downsampled to a (feature_resolution * feature_resolution) resolution for clustering.
            fusion_weights (list): Weights used for feature fusion. If None, a set of equal weights will be used.
            deviec (List[int]): GPU ids.
     """

    def __init__(self,
                 detector_class,
                 detector_config,
                 batch_size,
                 cluster_number = 4,
                 feature_resolution = 1, fusion_weights = None, device = None):

        self.detector_class = detector_class

        if type(detector_config) == dict:
            detector_config = EasyDict(detector_config)

        self.detector_config = detector_config
        self.batch_size = batch_size
        self.cluster_number = cluster_number
        if isinstance(feature_resolution, int):
            self.feature_resolution = [feature_resolution, feature_resolution]
        else:
            self.feature_resolution = feature_resolution
        self.fusion_weights = fusion_weights
        super().__init__(device)

    def create_tasks(self,
                        samples: List[HRSample],
                        indexes: List[MultiResolutionIndex],
                        **kwargs) -> List[Dict]:

        main_logger = logging.getLogger('main')
        main_logger.info(f'Start patch clustering, devices: {self.device}')

        embedding_pairs = patch_embedding(samples,
                                          indexes,
                                          self.detector_class,
                                          self.detector_config.patch,
                                          self.batch_size,
                                          self.feature_resolution,
                                          self.fusion_weights,
                                          self.device)

        index_embeddings = {}
        for _, index, embedding in embedding_pairs:
            if index not in index_embeddings:
                index_embeddings[index]= []
            index_embeddings[index].append(embedding)

        for index in index_embeddings:
            index_embeddings[index] = np.mean(np.stack(index_embeddings[index]),axis=0).reshape(-1)

        embeddings = np.stack([index_embeddings[key] for key in index_embeddings])
        embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=1).numpy()

        indexes = [ key for key in index_embeddings]

        clustering = KMeans(n_clusters=self.cluster_number, random_state=1)
        labels = clustering.fit_predict(embeddings)
        tasks = {}
        for index, label in zip(indexes, labels):
            if f'task_{label}' not in tasks:
                tasks[f'task_{label}'] = []
            tasks[f'task_{label}'].append(index)

        tasks = [{'name': task_name, 'indexes': tasks[task_name]} for task_name in tasks]
        tasks.sort(key= lambda item: int(item['name'].split('_')[-1]))
        return tasks


class ThresholdClusteringTaskGenerator(BaseTaskGenerator):
    r"""
        Cluster spatial patches with high visual similarity. Unlike `SpatialClusteringTaskGenerator`,
        this class does not require specifying the number of clusters in advance. Clustering is controlled by `cluster_threshold`:
        indexes with similarity ≥ cluster_threshold are grouped into the same cluster.
     """

    def __init__(self, detector_class, detector_config, batch_size,
                       cluster_threshold = 0.5, feature_resolution = 1, fusion_weights = None, device = None):

        self.detector_class = detector_class

        if type(detector_config) == dict:
            detector_config = EasyDict(detector_config)

        self.detector_config = detector_config
        self.batch_size = batch_size
        self.cluster_threshold = cluster_threshold
        if isinstance(feature_resolution, int):
            self.feature_resolution = [feature_resolution, feature_resolution]
        else:
            self.feature_resolution = feature_resolution
        self.fusion_weights = fusion_weights
        super().__init__(device)

    def create_tasks(self,
                        samples: List[HRSample],
                        indexes: List[MultiResolutionIndex],
                        **kwargs) -> List[Dict]:

        main_logger = logging.getLogger('main')
        main_logger.info(f'Start patch clustering, devices: {self.device}')

        embedding_pairs = patch_embedding(samples,
                                          indexes,
                                          self.detector_class,
                                          self.detector_config.patch,
                                          self.batch_size,
                                          self.feature_resolution,
                                          self.fusion_weights,
                                          self.device)

        index_embeddings = {}
        for _, index, embedding in embedding_pairs:
            if index not in index_embeddings:
                index_embeddings[index] = []
            index_embeddings[index].append(embedding)

        for index in index_embeddings:
            index_embeddings[index] = np.mean(np.stack(index_embeddings[index]), axis=0).reshape(-1)

        embeddings = np.stack([index_embeddings[key] for key in index_embeddings])
        embeddings = F.normalize(torch.tensor(embeddings),p=2,dim=1).numpy()

        indexes = [key for key in index_embeddings]

        clustering = DBSCAN(eps=self.cluster_threshold, min_samples=1)
        labels = clustering.fit_predict(embeddings)

        tasks = {}
        noise_tasks = []
        for index, label in zip(indexes, labels):
            if label == -1:
                noise_tasks.append(index)
            else:
                if f'task_{label}' not in tasks:
                    tasks[f'task_{label}'] = []
                tasks[f'task_{label}'].append(index)
        for noise_task in noise_tasks:
            tasks[f'task_{len(tasks)}'] = [noise_task]
        tasks = [{'name': task_name, 'indexes': tasks[task_name]} for task_name in tasks]
        tasks.sort(key= lambda item: int(item['name'].split('_')[-1]))
        return tasks


class RetrieveTaskGenerator(BaseTaskGenerator):
    r"""
        Clustering is based solely on patch similarity, ignoring spatial location. This class performs dynamic retrieval during inference.

        Args:
            detector_class (class): The provided detector class template, used to create detector instances across multiple processes.
                    Supported detector types are listed in `detectors.__init__`.
            detector_config: Detector configuration parameters.
            batch_size (int): Feature extraction Batch Size.
            cluster_number (int): Number of clusters.
            feature_resolution (int): The extracted multi-scale features are uniformly downsampled to a (feature_resolution * feature_resolution) resolution for clustering.
            save_root (str): If provided, the clustering results of the patches will be saved.
            fusion_weights (list): Weights used for feature fusion. If None, a set of equal weights will be used.
            deviec (List[int]): GPU ids.
     """
    def __init__(self, detector_class, detector_config, batch_size,
                       cluster_number = 4, feature_resolution = 1, save_root = None, fusion_weights = None, device = None):

        self.detector_class = detector_class
        if type(detector_config) == dict:
            detector_config = EasyDict(detector_config)

        self.detector_config = detector_config
        self.batch_size = batch_size
        self.cluster_number = cluster_number
        self.feature_resolution = feature_resolution
        self.save_root = save_root
        if isinstance(feature_resolution, int):
            self.feature_resolution = [feature_resolution, feature_resolution]
        else:
            self.feature_resolution = feature_resolution
        self.fusion_weights = fusion_weights
        super().__init__(device)


    def create_tasks(self,
                        samples: List[HRSample],
                        indexes: List[MultiResolutionIndex],
                        **kwargs) -> List[Dict]:

        main_logger = logging.getLogger('main')
        main_logger.info(f'Start Retrieve clustering, devices: {self.device}')

        embedding_pairs = patch_embedding(samples,
                                        indexes,
                                        self.detector_class,
                                        self.detector_config.patch,
                                        self.batch_size,
                                        self.feature_resolution,
                                        self.fusion_weights,
                                        self.device)

        embeddings = [pair[-1].reshape(-1) for pair in embedding_pairs]
        paths_and_indexes = [ pair[:-1] for pair in embedding_pairs]

        embeddings = np.stack(embeddings)
        embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=1).numpy()

        clustering = KMeans(n_clusters=self.cluster_number, random_state=1)
        labels = clustering.fit_predict(embeddings)

        centers = { f'retrieve_task_{label}': center for label, center in enumerate(clustering.cluster_centers_)}

        tasks = {}
        for path_and_index, label in zip(paths_and_indexes, labels):
            if f'retrieve_task_{label}' not in tasks:
                tasks[f'retrieve_task_{label}'] = []
            tasks[f'retrieve_task_{label}'].append(path_and_index)

        tasks = [{'name': task_name, 'paths_and_indexes': tasks[task_name],
                  'center': centers[task_name], 'feature_resolution':self.feature_resolution, 'fusion_weights': self.fusion_weights } for task_name in tasks]
        tasks.sort(key = lambda item: int(item['name'].split('_')[-1]))

        if self.save_root is not None:
            for task in tasks:
                task_root = os.path.join(self.save_root, task['name'])
                os.makedirs(task_root, exist_ok=True)
                for i, (path, index) in enumerate(tqdm.tqdm(task['paths_and_indexes'], desc=f'Save retrieve patch {task["name"]}')):
                    image = HRImage(image_path=path, is_mask=False, image_size = kwargs.get('image_size', None))
                    image.open()
                    patch = image[index.main_index]
                    image.close()
                    Image.fromarray(patch).save(os.path.join(task_root,f'patch_{i}.jpg'))
        return tasks