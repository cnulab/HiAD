import torch.multiprocessing as mp
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.neighbors import NearestCentroid
from hiad.utils.split_and_gather import MultiResolutionIndex


def embedding(samples, indexes, batch_size, model_manager, feature_resolution):

    if len(samples) == 0:
        return []

    emb_data = {index: [] for index in indexes}

    for sample in samples:
        for index in indexes:
            data = sample[index.main_index]
            if index.low_resolution_indexes is not None:
                for low_resolution_index in index.low_resolution_indexes:
                    data.add_low_resolution_images(low_resolution_index, sample.image)
            emb_data[index].append(data)

    detector = model_manager.get_detector(model_manager.get_device_task_names(gpu=True)[0])
    patch_embeddings = []

    for index in indexes:
        train_dataset = detector.create_dataset(emb_data[index], training=False, task_name=f'embedding_{str(index)}')
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = 0,
            pin_memory = True,
        )

        index_embeddings = []
        for data in train_dataloader:
            embeddings = detector.get_multi_resolution_fusion_embeddings(data)
            embeddings = [F.adaptive_avg_pool2d(embedding, feature_resolution) for embedding in embeddings]
            embeddings = torch.cat(embeddings, dim=1)
            index_embeddings.append(embeddings.cpu())

        index_embeddings = torch.cat(index_embeddings,dim=0).numpy()
        assert len(samples) == index_embeddings.shape[0]
        patch_embeddings.extend([[sample.get_image_path(), index, index_embeddings[i]] for i, sample in enumerate(samples)])

    return patch_embeddings


def update_paths_and_indexes_at_inferencer(tasks, samples, model_manager, batch_size):

    if 'feature_resolution' in tasks[0]:
        feature_resolution = tasks[0]['feature_resolution']
        for task in tasks:
            assert feature_resolution == task['feature_resolution']
    else:
        feature_resolution = 1

    indexes = []
    centers = []
    for task in tasks:
        centers.append(task['center'])
        for _, index in task['paths_and_indexes']:
            indexes.append(str(index))

    indexes = [MultiResolutionIndex.from_str(index) for index in set(indexes)]

    embedding_pairs = embedding(samples, indexes, batch_size, model_manager, feature_resolution)

    embeddings = [pair[-1].reshape(-1) for pair in embedding_pairs]
    paths_and_indexes = [pair[:-1] for pair in embedding_pairs]

    embeddings = np.stack(embeddings)
    embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=1).to(model_manager.gpu_device)
    centers = torch.tensor(np.stack(centers)).to(model_manager.gpu_device)
    dists = torch.cdist(embeddings, centers)
    labels = torch.argmin(dists, dim=-1).cpu().numpy()

    for i, task in enumerate(tasks):
        task['paths_and_indexes'] = [path_and_index for label, path_and_index in zip(labels, paths_and_indexes) if label == i]

    return tasks