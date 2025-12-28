import math
from typing import Union, List
import numpy as np
from PIL import Image
import cv2
import zmq
import time

from hiad.utils.split_and_gather import HRSample
from hiad.utils.base import split_list


def client_detection(samples: List[Union[str | np.ndarray | HRSample]],
                     image_size: Union[int, List] = None,
                     ip = '127.0.0,1', port = '1473', batch_size = -1, return_fps = False):

    images = []
    for sample in samples:
        if type(sample) == str:
            image = Image.open(sample).convert('RGB')
            if image_size is not None:
                image = image.resize(image_size if not isinstance(image_size, int) else (image_size, image_size), resample = Image.Resampling.BILINEAR)
            images.append(np.array(image))

        elif type(sample) == np.ndarray:
            if image_size is not None:
                image = cv2.resize(sample, image_size if not isinstance(image_size, int) else (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            images.append(image)

        elif type(sample) == HRSample:
            sample.image.image_size = image_size if not isinstance(image_size, int) else (image_size, image_size)
            sample.open()
            images.append(sample.image.image.copy())
            sample.close()

    if batch_size != -1:
        image_lists = split_list(images, math.ceil(len(images) / batch_size))
    else:
        image_lists = [images]

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://{}:{}".format(ip, port))
    image_scores = []
    anomaly_maps = []
    elapsed_s = 0

    for image_list in image_lists:
        socket.send_pyobj(image_list)
        start_time = time.perf_counter()
        result = socket.recv_pyobj()

        if result['status'] == 'success':
            end_time = time.perf_counter()
            elapsed_s += (end_time - start_time)
            image_scores.append(result['image_scores'])
            anomaly_maps.append(result['anomaly_maps'])
        else:
            raise RuntimeError(result['msg'])

    image_scores = np.concatenate(image_scores, axis=0)
    anomaly_maps = np.concatenate(anomaly_maps, axis=0)
    result = {"image_scores": image_scores, "anomaly_maps": anomaly_maps}
    if return_fps:
        result.update({"fps": round(len(samples) / elapsed_s, 2)})
    return result

