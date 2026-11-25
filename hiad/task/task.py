from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import torch
import tqdm
import numpy
import logging
import math
from sklearn.cluster import DBSCAN, KMeans
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
import os
from PIL import Image
from easydict import EasyDict
import torch.multiprocessing as mp

from hiad.task.utils import patch_embedding, update_paths_and_indexes
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
        mp.set_start_method('spawn', force=True)

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


def _held_out_nll_per_dim(train_embs, val_embs, eps=1e-4):
    """Compute per-dimension held-out NLL: (log-variance + Mahalanobis) / D."""
    n_val_total = sum(v.shape[0] for v in val_embs.values() if v.shape[0] > 0)
    if n_val_total == 0:
        return float('inf')

    nll = 0.0
    dim = 0
    for name, val_f in val_embs.items():
        if val_f.shape[0] == 0 or name not in train_embs:
            continue

        train_f = train_embs[name]
        dim = val_f.shape[1]  # Feature dimension (d)
        if train_f.shape[0] < 2:
            nll += (val_f.shape[0] / n_val_total) * dim * 10.0
            continue

        mu = train_f.mean(axis=0)
        var = train_f.var(axis=0, ddof=1)
        w = val_f.shape[0] / n_val_total

        nll += w * (np.sum(np.log(var + eps)) + np.mean(np.sum((val_f - mu) ** 2 / (var + eps), axis=1)))
    return nll / dim


def _spatial_penalty(tasks, indexes, is_ra):

    img_grids = {}
    task_name_to_id = {t['name']: i for i, t in enumerate(tasks)}

    grid = ImageIndexes2NdArray(indexes)
    idx_to_rc = {str(grid[r, c]): (r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])}

    for t in tasks:
        task_id = task_name_to_id[t['name']]
        if is_ra:
            for p, idx in t['paths_and_indexes']:
                if p not in img_grids:
                    img_grids[p] = np.zeros(grid.shape, dtype=int)

                if str(idx) in idx_to_rc:
                    r, c = idx_to_rc[str(idx)]
                    img_grids[p][r, c] = task_id

        else:
            for idx in t['indexes']:
                if "dummy" not in img_grids:
                    img_grids["dummy"] = np.zeros(grid.shape, dtype=int)

                if str(idx) in idx_to_rc:
                    r, c = idx_to_rc[str(idx)]
                    img_grids["dummy"][r, c] = task_id

    diff_edges, total_edges = 0, 0

    for g in img_grids.values():
        if g.shape[1] > 1:  # Horizontal edges
            diff_edges += np.sum(g[:, :-1] != g[:, 1:])
            total_edges += g.shape[0] * (g.shape[1] - 1)

        if g.shape[0] > 1:  # Vertical edges
            diff_edges += np.sum(g[:-1, :] != g[1:, :])
            total_edges += (g.shape[0] - 1) * g.shape[1]

    f_spatial = diff_edges / max(total_edges, 1)
    return f_spatial



class AutoTaskGenerator(BaseTaskGenerator):
    r"""
    Selects among {NA, SCA, RA} via multi-criteria evaluation:
    """
    def __init__(self,
                 detector_class,
                 detector_config,
                 batch_size,
                 cluster_number=4,
                 feature_resolution=1,
                 fusion_weights=None, device=None, val_ratio=0.2):

        self.detector_class = detector_class
        if isinstance(detector_config, dict):
            detector_config = EasyDict(detector_config)

        self.detector_config = detector_config
        self.batch_size = batch_size
        self.cluster_number = cluster_number

        if isinstance(feature_resolution, int):
            self.feature_resolution = [feature_resolution, feature_resolution]
        else:
            self.feature_resolution = feature_resolution

        self.fusion_weights = fusion_weights
        self.val_ratio = val_ratio
        super().__init__(device)

    def _collect(self, tasks, pairs, is_ra=False):
        if is_ra:
            k2t = {}
            for t in tasks:
                for p, i in t['paths_and_indexes']:
                    k2t[f'{p}_{i}'] = t['name']
            e = {t['name']: [] for t in tasks}
            for p, i, em in pairs:
                k = f'{p}_{i}'
                if k in k2t: e[k2t[k]].append(em.reshape(-1))
        else:
            i2t = {}
            for t in tasks:
                for i in t['indexes']: i2t[i] = t['name']
            e = {t['name']: [] for t in tasks}
            for _, i, em in pairs:
                if i in i2t: e[i2t[i]].append(em.reshape(-1))

        return {n: np.stack(v) for n, v in e.items() if v}


    def _get_na_groups(self):
        s = int(math.sqrt(self.cluster_number))
        for i in range(s, 0, -1):
            if self.cluster_number % i == 0:
                return sorted((i, self.cluster_number // i))
        return (1, self.cluster_number)


    def create_tasks(self, samples, indexes, **kwargs):
        logger = logging.getLogger('main')

        train_samples, val_samples = train_test_split(samples, test_size=self.val_ratio)

        train_pairs = patch_embedding(
            train_samples, indexes, self.detector_class,
            self.detector_config.patch, self.batch_size,
            self.feature_resolution, self.fusion_weights, self.device)

        val_pairs = patch_embedding(
            val_samples, indexes, self.detector_class,
            self.detector_config.patch, self.batch_size,
            self.feature_resolution, self.fusion_weights, self.device)

        N = len(train_pairs) + len(val_pairs)
        d = train_pairs[0][-1].reshape(-1).shape[0]

        logger.info(f'AutoTaskGenerator: N={N}, d={d}, M={self.cluster_number}')

        cands = {}

        cands['NA'] = NeighborhoodTaskGenerator(
            *self._get_na_groups(), device=self.device
        ).create_tasks(train_samples, indexes)

        cands['SCA'] = SpatialClusteringTaskGenerator(
            self.detector_class, self.detector_config, self.batch_size,
            cluster_number=self.cluster_number,
            feature_resolution=self.feature_resolution,
            fusion_weights=self.fusion_weights, device=self.device
        ).create_tasks(train_samples, indexes=indexes, embedding_pairs=train_pairs)

        cands['RA'] = RetrieveTaskGenerator(
            self.detector_class, self.detector_config, self.batch_size,
            cluster_number=self.cluster_number,
            feature_resolution=self.feature_resolution,
            fusion_weights=self.fusion_weights, device=self.device
        ).create_tasks(train_samples, indexes=indexes, embedding_pairs=train_pairs)

        nll_scores = {}
        spatial_scores = {}

        for name, tasks in cands.items():
            is_ra = (name == 'RA')
            tr_e = self._collect(tasks, train_pairs, is_ra)
            if is_ra:
                tasks = update_paths_and_indexes(
                    tasks,
                    val_samples,
                    self.detector_class,
                    self.detector_config,
                    self.batch_size,
                    self.device,
                    embedding_pairs=val_pairs)
                va_e = self._collect(tasks, val_pairs, is_ra)
            else:
                va_e = self._collect(tasks, val_pairs, False)

            nll_scores[name] = _held_out_nll_per_dim(tr_e, va_e)
            spatial_scores[name] = _spatial_penalty(tasks, indexes, is_ra)

        final_scores = {key: nll_scores[key] + 0.5 * spatial_scores[key] for key in nll_scores}

        logger.info(f'  {"":>4s}  {"NLL":>10s} {"SP":>10s} {"Score":>10s}')

        for name in final_scores:
            logger.info(
                f'  {name:>4s}  {nll_scores[name]:>+10.2f} {spatial_scores[name]:>+10.2f}'
                f'{final_scores[name]:>10.3f}'
            )

        best = min(final_scores, key=final_scores.get)
        
        epsilon = 0.05
        best = 'SCA' if best == 'RA' and final_scores['SCA'] - final_scores['RA'] <= epsilon else best
        
        logger.info(f'AutoTaskGenerator: selected {best} (Score={final_scores[best]:.3f})')

        if best == 'NA':
            return NeighborhoodTaskGenerator(
               *self._get_na_groups(), device=self.device
           ).create_tasks(samples, indexes)
        elif best == 'SCA':
            return SpatialClusteringTaskGenerator(
                self.detector_class, self.detector_config, self.batch_size,
                cluster_number=self.cluster_number,
                feature_resolution=self.feature_resolution,
                fusion_weights=self.fusion_weights, device=self.device
            ).create_tasks(samples, indexes=indexes, embedding_pairs=train_pairs + val_pairs)
        elif best == 'RA':
            return RetrieveTaskGenerator(
                self.detector_class, self.detector_config, self.batch_size,
                cluster_number=self.cluster_number,
                feature_resolution=self.feature_resolution,
                fusion_weights=self.fusion_weights, device=self.device
            ).create_tasks(samples, indexes=indexes, embedding_pairs=train_pairs + val_pairs)



