import os
import random
import torch
import copy
import numpy
import torch.nn.functional as F

from hiad.utils.base import create_logger, set_seed
from hiad.syn import BaseAnomalySynthesizer

def train_in_device(gpu_id,
                    detector_class,
                    config,
                    train_samples,
                    task_in_device,
                    batch_size,
                    checkpoint_root,
                    log_root,
                    seed,
                    val_config = None,
                    fusion_weights = None,
                    early_stop_epochs = -1):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda")
    set_seed(seed)

    logger = create_logger(
                f"train_logger_device{gpu_id}", os.path.join(log_root, f"train_log_device{gpu_id}.log")
            )

    logger.info(f'Device {gpu_id} start training')
    logger.info(f'Task Num: {len(task_in_device)}, Task list is:')

    for index, task in enumerate(task_in_device):
        if task['name'].startswith('retrieve'):
            logger.info(f'[{index+1}/{len(task_in_device)}] Task name: {task["name"]}, Patch Num: {len(task["paths_and_indexes"])}')
        elif task['name'] == 'thumbnail':
            logger.info(f'[{index+1}/{len(task_in_device)}] Task name: {task["name"]}')
        else:
            logger.info(f'[{index+1}/{len(task_in_device)}] Task name: {task["name"]}, Patch Num: {len(task["indexes"])}')

    for task in task_in_device:  # a single train
        task_name = task['name']
        logger.info("")
        logger.info(f'Task {task["name"]} loading images')

        if task_name.startswith('retrieve'):
            paths_and_indexes = task['paths_and_indexes']
            sample_map = {}
            for path, index in paths_and_indexes:
                if path not in sample_map:
                    sample_map[path] = []
                sample_map[path].append(index)
            lr_sampls = []
            detector = detector_class(**copy.deepcopy(config.patch), logger=logger, device=device, seed=seed,
                                      fusion_weights = fusion_weights, early_stop_epochs = early_stop_epochs)

            for i, sample in enumerate(train_samples):
                if len(train_samples) > 10 and (i + 1) % int(len(train_samples) * 0.1) == 0:
                    logger.info(f'{int((i + 1) / int(len(train_samples) * 0.1) * 10)}% of data loaded')
                if sample.get_image_path() in sample_map:
                    sample.open()
                    for index in sample_map[sample.get_image_path()]:
                        data = sample[index.main_index]
                        if index.low_resolution_indexes is not None:
                            for low_resolution_index in index.low_resolution_indexes:
                                data.add_low_resolution_images(low_resolution_index, sample.image)
                        lr_sampls.append(data)
                    sample.close()

        elif task_name == 'thumbnail':
            thumbnail_config = copy.deepcopy(config.thumbnail)
            thumbnail_config.patch_size = thumbnail_config.pop('thumbnail_size')
            detector = detector_class(**thumbnail_config, logger=logger, device=device, seed=seed,
                                      fusion_weights = fusion_weights,
                                      early_stop_epochs = early_stop_epochs)
            lr_sampls = []
            for i, sample in enumerate(train_samples):
                if len(train_samples) > 10 and (i + 1) % int(len(train_samples) * 0.1) == 0:
                    logger.info(f'{int((i + 1) / int(len(train_samples) * 0.1) * 10)}% of data loaded')
                sample.open()
                lr_sampls.append(sample.down_sampling_to_LR(task['thumbnail_size']))
                sample.close()
        else:
            indexes = task['indexes']
            lr_sampls = []
            detector = detector_class(**copy.deepcopy(config.patch), logger=logger, device=device, seed=seed,
                                      fusion_weights = fusion_weights,
                                      early_stop_epochs = early_stop_epochs)
            for i, sample in enumerate(train_samples):
                if len(train_samples) > 10 and (i + 1) % int(len(train_samples) * 0.1) == 0:
                    logger.info(f'{int((i + 1) / int(len(train_samples) * 0.1) * 10)}% of data loaded')
                sample.open()
                for index in indexes:
                    data = sample[index.main_index]
                    if index.low_resolution_indexes is not None:
                        for low_resolution_index in index.low_resolution_indexes:
                            data.add_low_resolution_images(low_resolution_index, sample.image)
                    lr_sampls.append(data)
                sample.close()

        if val_config is not None:
            val_sample_num = int(len(lr_sampls) * val_config['val_sample_ratio'])
            val_sample_indexes = random.sample(list(range(len(lr_sampls))), val_sample_num)
            train_lr_samples = [lr_sampls[i] for i in range(len(lr_sampls)) if i not in val_sample_indexes]
            val_lr_samples = [lr_sampls[i] for i in range(len(lr_sampls)) if i in val_sample_indexes]
            if 'max_patch_number' in val_config and len(val_lr_samples) > val_config['max_patch_number']:
                val_lr_samples = random.sample(val_lr_samples, val_config['max_patch_number'])
            anomaly_syn_fn = val_config['anomaly_syn_fn']

            assert isinstance(anomaly_syn_fn, BaseAnomalySynthesizer)
            val_lr_samples = [anomaly_syn_fn.anomaly_synthesize(sample) for sample in val_lr_samples]

            assert numpy.min([sample.label for sample in val_lr_samples]) == 0 and \
                   numpy.max([sample.label for sample in val_lr_samples]) == 1

            train_dataset = detector.create_dataset(train_lr_samples, training=True, task_name=task_name)

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers = 0,
                pin_memory = True,
                drop_last = True
            )

            val_dataset = detector.create_dataset(val_lr_samples, training=False, task_name="val_" + task_name)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle= False,
                num_workers = 0,
                pin_memory = True,
            )
            evaluators = val_config['evaluators'] if 'evaluators' in val_config else None
        else:
            train_dataset = detector.create_dataset(lr_sampls, training=True, task_name=task_name)
            train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size = batch_size,
                    shuffle = True,
                    num_workers = 0,
                    pin_memory = True,
                    drop_last=True
            )
            val_dataloader = None
            evaluators = None

        logger.info(f'Task {task["name"]} start training')
        logger.info(f'train dataset len is: {len(train_dataloader.dataset)}')
        if val_dataloader is not None:
            logger.info(f'val dataset len is: {len(val_dataloader.dataset)}')

        checkpoint_path = os.path.join(checkpoint_root,f'{task_name}_weight.pkl')
        is_saved = detector.train_step(train_dataloader, task_name, checkpoint_path,
                                       val_dataloader = val_dataloader,
                                       evaluators = evaluators)
        if not is_saved:
            detector.save_checkpoint(checkpoint_path)
        logger.info(f'Task {task["name"]} end training, the checkpoint is saved as {checkpoint_path}')
    logger.info('All tasks are done.')


def test_in_device( gpu_id,
                    detector_class,
                    config,
                    test_samples,
                    task_in_device,
                    batch_size,
                    checkpoint_root,
                    log_root,
                    seed):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda")
    set_seed(seed)

    logger = create_logger(
                f"inference_logger_device{gpu_id}", os.path.join(log_root, f"inference_log_device{gpu_id}.log")
            )

    logger.info(f'Device {gpu_id} start inference')
    logger.info(f'Task Num: {len(task_in_device)}, Task list is:')

    for index, task in enumerate(task_in_device):
        if task['name'].startswith('retrieve'):
            logger.info(f'[{index+1}/{len(task_in_device)}] Task name: {task["name"]}, Patch Num: {len(task["paths_and_indexes"])}')
        elif task['name'] == 'thumbnail':
            logger.info(f'[{index+1}/{len(task_in_device)}] Task name: {task["name"]}')
        else:
            logger.info(f'[{index+1}/{len(task_in_device)}] Task name: {task["name"]}, Patch Num: {len(task["indexes"])}')

    device_results = {sample.image.image_path: {"patch_list": [], "index_list": []} for sample in test_samples}
    logger.info(f'Start loading data')

    test_data = {}

    for i, sample in enumerate(test_samples):
        if len(test_samples) > 10 and (i + 1) % int(len(test_samples) * 0.1) == 0:
            logger.info(f'{int((i + 1) / int(len(test_samples) * 0.1) * 10)}% of data loaded')

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
                    sample.open()
                    indexes = sample_map[sample.get_image_path()]
                    for index in indexes:
                        data = sample[index.main_index]
                        if index.low_resolution_indexes is not None:
                            for low_resolution_index in index.low_resolution_indexes:
                                data.add_low_resolution_images(low_resolution_index, sample.image)
                        test_data[task_name].append([sample.get_image_path(), index, data])
                    sample.close()

            elif task_name=='thumbnail':
                if task_name not in test_data:
                    test_data[task_name] = []
                sample.open()
                test_data[task_name].append(sample.down_sampling_to_LR(task['thumbnail_size']))
                sample.close()
            else:
                if task_name not in test_data:
                    test_data[task_name] = {}
                sample.open()
                indexes = task['indexes']
                for index in indexes:
                    if index not in test_data[task_name]:
                        test_data[task_name][index] = []
                    data = sample[index.main_index]
                    if index.low_resolution_indexes is not None:
                        for low_resolution_index in index.low_resolution_indexes:
                            data.add_low_resolution_images(low_resolution_index, sample.image)
                    test_data[task_name][index].append(data)
                sample.close()

    logger.info(f'End loading data')

    for task in task_in_device:
        task_name = task['name']
        logger.info(f'Task {task_name} start inference')

        if task_name.startswith('retrieve'):

            detector = detector_class(**copy.deepcopy(config.patch), logger=logger, device=device, seed=seed)
            checkpoint_path = os.path.join(checkpoint_root, f'{task_name}_weight.pkl')
            logger.info(f'Load checkpoint from {checkpoint_path}')
            detector.load_checkpoint(checkpoint_path)

            paths = [path for path, index, data in test_data[task_name]]
            indexes = [index for path, index, data in test_data[task_name]]
            lr_sampls = [data for path, index, data in test_data[task_name]]

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
            thumbnail_config = copy.deepcopy(config.thumbnail)
            thumbnail_config.patch_size = thumbnail_config.pop('thumbnail_size')
            detector = detector_class(**thumbnail_config, logger=logger, device=device, seed=seed)
            checkpoint_path = os.path.join(checkpoint_root, f'{task_name}_weight.pkl')
            logger.info(f'Load checkpoint from {checkpoint_path}')
            detector.load_checkpoint(checkpoint_path)

            lr_sampls = test_data[task_name]
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
            detector = detector_class(**copy.deepcopy(config.patch), logger=logger, device=device, seed=seed)
            checkpoint_path = os.path.join(checkpoint_root, f'{task_name}_weight.pkl')
            logger.info(f'Load checkpoint from {checkpoint_path}')
            detector.load_checkpoint(checkpoint_path)

            for index in indexes:
                lr_sampls = test_data[task_name][index]
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

        logger.info(f'Task {task_name} end inference')

    logger.info(f'Device {gpu_id} end inference')
    return device_results


def compute_metrics_in_device(gpu_id, clsnames, evaluators, prediction_masks, gt_masks, gt_labels, prediction_scores, clsname_list):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda")
    scores = []

    for clsname in clsnames:
        score = {'clsname': clsname}

        cls_prediction_masks = numpy.stack( [prediction_masks[i].numpy() if isinstance(prediction_masks, torch.Tensor) else prediction_masks[i]
                                             for i, cls in enumerate(clsname_list) if cls == clsname])

        cls_gt_masks = numpy.stack([gt_masks[i].numpy() if isinstance(gt_masks, torch.Tensor) else gt_masks[i]
                                    for i, cls in enumerate(clsname_list) if cls == clsname])

        cls_gt_labels = numpy.stack([gt_labels[i] for i, cls in enumerate(clsname_list) if cls == clsname])

        cls_prediction_scores = numpy.stack(
            [prediction_scores[i] for i, cls in enumerate(clsname_list) if cls == clsname])

        for evaluator_fn in evaluators:
            score.update(evaluator_fn(prediction_masks=cls_prediction_masks,
                                           gt_masks=cls_gt_masks,
                                           prediction_scores=cls_prediction_scores,
                                           gt_labels=cls_gt_labels, device=device))
        scores.append(score)
    return scores



def embedding_in_device(gpu_id,
                        detector_class,
                        detector_config,
                        samples,
                        indexes,
                        batch_size,
                        fusion_weights,
                        feature_resolution):

    if len(samples) == 0:
        return []

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda")

    emb_data = { index: [] for index in indexes}

    for sample in samples:
        sample.open()
        for index in indexes:
            data = sample[index.main_index]
            if index.low_resolution_indexes is not None:
                for low_resolution_index in index.low_resolution_indexes:
                    data.add_low_resolution_images(low_resolution_index, sample.image)
            emb_data[index].append(data)
        sample.close()

    detector = detector_class(**copy.deepcopy(detector_config), fusion_weights=fusion_weights, logger=None, device=device)
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
            embeddings = torch.cat(embeddings,dim=1)
            index_embeddings.append(embeddings.cpu())

        index_embeddings = torch.cat(index_embeddings,dim=0).numpy()
        assert len(samples) == index_embeddings.shape[0]
        patch_embeddings.extend([ [sample.get_image_path(), index, index_embeddings[i]]  for i, sample in enumerate(samples)])
    return patch_embeddings