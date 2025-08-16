import torch.multiprocessing as mp
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.neighbors import NearestCentroid

from hiad.utils.device import embedding_in_device
from hiad.utils.split_and_gather import MultiResolutionIndex
from hiad.utils.base import split_list

def patch_embedding(samples,
                    indexes,
                    detector_class,
                    detector_config,
                    batch_size,
                    feature_resolution,
                    fusion_weights,
                    devices):

    num_device = len(devices)
    samples_in_device = split_list(samples, num_device)
    process_pool = mp.Pool(processes=num_device)
    embeddings = []

    for gpu_id, samples_ in zip(devices, samples_in_device):
        embedding = process_pool.apply_async(embedding_in_device, args=(gpu_id, detector_class,
                                                                        detector_config,
                                                                        samples_, indexes, batch_size,
                                                                        fusion_weights,
                                                                        feature_resolution))
        embeddings.append(embedding)

    process_pool.close()
    process_pool.join()
    patch_embeddings = []
    for embedding in embeddings:
        embedding = embedding.get()
        patch_embeddings.extend(embedding)
    return patch_embeddings


def update_paths_and_indexes(
                    tasks,
                    samples,
                    detector_class,
                    detector_config,
                    batch_size,
                    devices):

    if 'feature_resolution' in tasks[0]:
        feature_resolution = tasks[0]['feature_resolution']
        for task in tasks:
            assert feature_resolution == task['feature_resolution']
    else:
        feature_resolution = 1

    if 'fusion_weights' in tasks[0]:
        fusion_weights = tasks[0]['fusion_weights']
    else:
        fusion_weights = None

    indexes = []
    centers = []
    for task in tasks:
        centers.append(task['center'])
        for _, index in task['paths_and_indexes']:
            indexes.append(str(index))

    indexes = [MultiResolutionIndex.from_str(index) for index in set(indexes)]

    embedding_pairs = patch_embedding(samples,
                                    indexes,
                                    detector_class,
                                    detector_config,
                                    batch_size,
                                    feature_resolution,
                                    fusion_weights,
                                    devices)

    embeddings = [pair[-1].reshape(-1) for pair in embedding_pairs]
    paths_and_indexes = [pair[:-1] for pair in embedding_pairs]

    embeddings = np.stack(embeddings)
    embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=1).numpy()

    clf = NearestCentroid()
    clf.centroids_ = np.stack(centers)
    clf.classes_ = np.arange(np.stack(centers).shape[0])
    labels = clf.predict(embeddings)

    for i, task in enumerate(tasks):
        task['paths_and_indexes'] = [ path_and_index  for label, path_and_index in  zip(labels, paths_and_indexes) if label == i]
    return tasks