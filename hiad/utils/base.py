import copy
import os
import json
import logging
import numpy
import numpy as np
import torch
import random
from typing import List, Dict
import torch.multiprocessing as mp
import tabulate
from .split_and_gather import MultiResolutionIndex


def read_meta_file(path:str):
    with open(path,'r+') as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]


def split_list(lst, n):
    out = [[] for _ in range(n)]
    cur = 0
    for item in lst:
        out[cur].append(item)
        cur += 1
        if cur == len(out):
            cur = 0
    return out


def create_logger(name, log_file, print_console=False, level=logging.INFO):
    logging.getLogger().handlers.clear()
    log = logging.getLogger(name)
    log.handlers.clear()
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d]%(message)s"
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)

    if print_console:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)
    return log


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def get_avg_score(scores):
    return numpy.mean([scores[key] for key in scores if key.find('threshold') == -1])


def save_tasks_config(tasks, save_path: str):
    task_list = []
    for task in tasks:
        if task['name']=='thumbnail':
            task_list.append({
                "name": task['name'],
                "thumbnail_size": task['thumbnail_size']
            })
        elif task['name'].startswith('retrieve'):
             center_save_path = os.path.join(os.path.dirname(save_path), f"{task['name']}_center.npy")
             np.save(center_save_path, task['center'])
             task_list.append({
                 "name": task['name'],
                 "paths_and_indexes": [[path, str(index)] for path, index in task['paths_and_indexes']],
                 'center': center_save_path ,
                 'feature_resolution': task['feature_resolution'],
                 'fusion_weights': task['fusion_weights'],
             })
        else:
            task_list.append({
                "name": task['name'],
                "indexes": [str(index) for index in task['indexes']]
            })

    with open(save_path, 'w+') as f:
        json.dump(task_list, f)


def load_tasks(load_path: str):
    with open(load_path,'r+') as f:
        tasks = json.load(f)
    for task in tasks:
        if task['name'].startswith('retrieve'):
            task['center'] = np.load(task['center'])
            task['paths_and_indexes'] = [ [path, MultiResolutionIndex.from_str(index)] for path, index in task['paths_and_indexes']]
        elif task['name'] != 'thumbnail':
            task['indexes'] = [MultiResolutionIndex.from_str(index) for index in task['indexes']]
    return tasks


def ImageIndexes2NdArray(indexes: List[MultiResolutionIndex]):
    IndexesMap = {}
    for index in indexes:
        if index.main_index.y not in IndexesMap:
            IndexesMap[index.main_index.y] = [ ]
        IndexesMap[index.main_index.y].append(index)

    for key in IndexesMap:
        IndexesMap[key].sort(key=lambda item:item.main_index.x)

    IndexesList = [IndexesMap[key] for key in IndexesMap]
    IndexesList.sort(key= lambda items: items[0].main_index.y)
    return numpy.array(IndexesList)


def create_share_array(src_array):
    tensor = torch.from_numpy(src_array)
    shared_tensor = tensor.share_memory_()
    return shared_tensor


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )



def PrintTasks(tasks: List[Dict]):

    tasks = [task for task in tasks if task['name'] != 'thumbnail']
    is_kv_task = tasks[0]['name'].startswith('retrieve')

    if not is_kv_task:
        IndexesMap = {}
        IndexesList = []

        for task in tasks:
            task_name = task['name']
            indexes = task['indexes']
            IndexesList.extend(indexes)
            for index in indexes:
                IndexesMap[index.main_index] = task_name

        IndexesArray = ImageIndexes2NdArray(IndexesList)

        print(f"Row: {IndexesArray.shape[0]}, Column: {IndexesArray.shape[1]}, Patch Number: {IndexesArray.shape[0]*IndexesArray.shape[1]}, Task Number: {len(tasks)}")
        for row_index in range(IndexesArray.shape[0]):
            print('|',end=' ')
            for col_index in range(IndexesArray.shape[1]):
                task_name = IndexesMap[IndexesArray[row_index][col_index].main_index]
                print(f'{task_name} |',end=' ')
            print()
    else:
        kv_info =  {task['name']: len(task['paths_and_indexes']) for task in tasks}
        total_patch = sum([kv_info[name] for name in kv_info])
        print('-------')
        for name in kv_info:
            print(f'Task Name: {name}, Percentage of patches: { (kv_info[name]/total_patch * 100):.2f}%')
        print('-------')
