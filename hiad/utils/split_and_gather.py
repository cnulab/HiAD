from typing import Union, List
import copy
import gc
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Union, List
import json
import cv2


@dataclass
class HRImageIndex:
    x: int
    y: int
    width : int
    height : int

    def __str__(self):
        d = {'x': self.x, 'y' : self.y, 'width' : self.width, 'height' : self.height}
        return json.dumps(d)

    def __hash__(self):
        return str(self).__hash__()

    def __eq__(self, other):
        return str(self) == str(other)

    def todict(self):
        return {'x': self.x, 'y' : self.y, 'width' : self.width, 'height' : self.height}

    @staticmethod
    def from_str(s):
        d = json.loads(s)
        return HRImageIndex(x=d['x'], y=d['y'], width=d['width'], height=d['height'])


class MultiResolutionIndex:

    def __init__(self,
                 main_index: HRImageIndex,
                 low_resolution_indexes: List[HRImageIndex] = None):
        self.main_index = main_index
        self.low_resolution_indexes = low_resolution_indexes

    def add_low_resolution_index(self, candidate_indexes):
        main_x_start = self.main_index.x
        main_y_start = self.main_index.y
        main_x_end = self.main_index.x + self.main_index.width
        main_y_end = self.main_index.y + self.main_index.height
        for index in candidate_indexes:
            x_start = index.x
            y_start = index.y
            x_end = index.x + index.width
            y_end = index.y + index.height
            if main_x_start >= x_start and main_y_start >= y_start and main_x_end <= x_end and main_y_end <= y_end:
                if self.low_resolution_indexes is None:
                    self.low_resolution_indexes = []
                self.low_resolution_indexes.append(index)
                return True
        return False

    def __str__(self):

        d = {'main_index': self.main_index.todict(),
             'low_resolution_indexes': [index.todict() for index in self.low_resolution_indexes] if self.low_resolution_indexes is not None else None}
        return json.dumps(d)

    def __hash__(self):
        return str(self).__hash__()

    def __eq__(self, other):
        return str(self) == str(other)

    @staticmethod
    def from_str(s):
        d = json.loads(s)
        return MultiResolutionIndex(main_index=HRImageIndex(x=d['main_index']['x'],
                                                              y=d['main_index']['y'],
                                                              width=d['main_index']['width'],
                                                              height=d['main_index']['height']),
                                    low_resolution_indexes =[HRImageIndex(x=index['x'],
                                                              y=index['y'],
                                                              width=index['width'],
                                                              height=index['height'])
                                for index in d['low_resolution_indexes']] if d['low_resolution_indexes'] is not None else None)


class HRImage:
    def __init__(self,
                 image_path: str,
                 image_size: Union[int, List] = None,
                 is_mask: bool = False):

        self.image_path = image_path
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.is_mask = is_mask
        self.image = None

    def open(self):
        if self.is_mask:
            self.image = Image.open(self.image_path).convert('L')
            if self.image_size is not None:
                self.image = self.image.resize(self.image_size, resample=Image.Resampling.NEAREST)
            self.image = np.array(self.image)
        else:
            self.image = Image.open(self.image_path).convert('RGB')
            if self.image_size is not None:
                self.image = self.image.resize(self.image_size, resample=Image.Resampling.BILINEAR)
            self.image = np.array(self.image)

    def close(self):
        del self.image
        gc.collect()
        self.image = None

    def size(self):
        assert self.image is not None
        shape = self.image.shape[:-1][::-1]
        return shape

    def resize(self, image_size: Union[int, List[int]]):
        assert self.image is not None
        if self.is_mask:
            if isinstance(image_size, int):
                image = Image.fromarray(self.image).resize((image_size, image_size),resample=Image.Resampling.NEAREST)
            else:
                image = Image.fromarray(self.image).resize(image_size, resample=Image.Resampling.NEAREST)
        else:
            if isinstance(image_size, int):
                image = Image.fromarray(self.image).resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
            else:
                image = Image.fromarray(self.image).resize(image_size, resample=Image.Resampling.BILINEAR)
        return np.array(image)

    def __getitem__(self, item: HRImageIndex):
        assert self.image is not None
        x_start = item.x
        x_end = item.x + item.width
        y_start = item.y
        y_end = item.y + item.height
        if self.image.ndim == 3:
            image_patch = self.image[y_start:y_end, x_start: x_end, :]
        else:
            image_patch = self.image[y_start:y_end, x_start: x_end]
        return copy.deepcopy(image_patch)

@dataclass
class LRPatch:
    image: np.ndarray
    mask: np.ndarray = None
    foreground: np.ndarray = None
    label: int = None
    label_name: str = None
    clsname: str = None # image category
    main_index: HRImageIndex= None
    low_resolution_images: List[np.ndarray] = None
    low_resolution_indexes: List[HRImageIndex] = None

    def add_low_resolution_images(self, low_resolution_index: HRImageIndex, image: HRImage):
        H_main, W_main = self.image.shape[:2]
        low_resolution_image = image[low_resolution_index]
        H_LOW, W_LOW = low_resolution_image.shape[:2]
        low_resolution_image = cv2.resize(low_resolution_image, (H_main, W_main))

        new_low_resolution_index = HRImageIndex( x = int((self.main_index.x - low_resolution_index.x) / W_LOW * W_main),
                                                 y = int((self.main_index.y - low_resolution_index.y) / H_LOW * H_main),
                                                 width = int(self.main_index.width / low_resolution_index.width * W_main),
                                                 height = int(self.main_index.height / low_resolution_index.height * H_main))

        if self.low_resolution_indexes is None:
            self.low_resolution_indexes = []
        if self.low_resolution_images is None:
            self.low_resolution_images = []

        self.low_resolution_images.append(low_resolution_image)
        self.low_resolution_indexes.append(new_low_resolution_index)


class HRSample:
    def __init__(self,
                 image: Union[str, HRImage],
                 mask: Union[str, HRImage] = None,
                 foreground: Union[str, HRImage] = None,
                 image_size: Union[int, List] = None,
                 label: int = None,
                 label_name: str = None,
                 clsname: str = None):

        self.image = image
        self.mask = mask
        self.foreground = foreground
        self.label = label
        self.label_name = label_name
        self.clsname = clsname

        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

        if isinstance(self.image, str):
            self.image = HRImage(image_path=self.image, is_mask=False, image_size=self.image_size)

        if self.mask is not None and isinstance(self.mask, str):
            self.mask = HRImage(image_path=self.mask, is_mask=True, image_size=self.image_size)

        if self.foreground is not None and isinstance(self.foreground, str):
            self.foreground = HRImage(image_path=self.foreground, is_mask=True, image_size=self.image_size)

    def __getitem__(self, item: HRImageIndex) -> LRPatch:
        assert self.image.image is not None
        value = {"image": self.image[item]}

        if self.mask is not None:
            value["mask"] = self.mask[item]

        if self.foreground is not None:
            value['foreground'] = self.foreground[item]

        if self.clsname is not None:
            value['clsname'] = self.clsname

        if self.label is not None:
            value['label'] = self.label

        if self.label_name is not None:
            value['label_name'] = self.label_name

        return LRPatch(**value, main_index = item)


    def open(self):
        self.image.open()
        if self.mask is not None:
            self.mask.open()
        if self.foreground is not None:
            self.foreground.open()

    def close(self):
        self.image.close()
        if self.mask is not None:
            self.mask.close()
        if self.foreground is not None:
            self.foreground.close()

    def down_sampling_to_LR(self, image_size: Union[int, List[int]]) -> LRPatch:
        if self.image.image is None:
            self.open()
        sample = LRPatch(image= self.image.resize(image_size))
        sample.mask = self.mask.resize(image_size) if self.mask is not None else None
        sample.foreground = self.foreground.resize(image_size) if self.foreground is not None else None
        sample.label = self.label
        sample.clsname = self.clsname
        sample.label_name = self.label_name
        return sample

    def get_image_path(self):
        return self.image.image_path


def MultiResolutionHRImageSpliter(image_size: Union[int, List],
                                    patch_size: Union[int, List],
                                    ds_factors: List[int] = None,
                                    stride: Union[int, List] = None) -> List[MultiResolutionIndex]:
    r"""
    :param image_size: Detection resolution: images will be resized to this size for detection
    :param patch_size: Patch size: size of the image patches after cropping
    :param ds_factors: Downsampling ratio used for feature fusion
    :param stride: Stride size: set to -1 to use the same value as the patch size
    :return: List of created indexes
    """

    if ds_factors is None:
        ds_factors = [0]

    ds_factors.sort()
    ds_factors = [2 ** factor for factor in ds_factors]
    main_factor = ds_factors[0]

    if isinstance(image_size, int):
        image_size = [image_size, image_size]

    if isinstance(patch_size, int):
        patch_size = [patch_size, patch_size]

    if stride is not None and isinstance(stride, int):
        stride = [stride, stride]

    main_patch_size = [v * main_factor for v in patch_size]
    main_stride = None if stride is None else [v * main_factor for v in stride]

    main_indexes = HRImageSpliter(image_size, main_patch_size, main_stride)
    indexes = [MultiResolutionIndex(main_index = index) for index in main_indexes]
    for factor in ds_factors[1:]:
        patch_size_multi_resolution = [v * factor for v in patch_size]
        stride_scale_multi_resolution = None if stride is None else [v * factor for v in stride]
        low_resolution_indexes = HRImageSpliter(image_size, patch_size_multi_resolution, stride_scale_multi_resolution)
        for index in indexes:
            assert index.add_low_resolution_index(low_resolution_indexes)
    return indexes


def HRImageSpliter(image_size: Union[int, List],
                   patch_size: Union[int, List],
                   stride: Union[int, List] = None) -> List[HRImageIndex]:

    def extract_starts(image_size, patch_size, stride):
        if image_size <= patch_size:
            starts = [0, ]
        else:
            starts = list(range(0, image_size, stride))
            for i in range(len(starts)):
                if starts[i] + patch_size > image_size:
                    starts[i] = image_size - patch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    if isinstance(image_size, int):
        image_width, image_height = image_size, image_size
    else:
        image_width, image_height = image_size[0], image_size[1]

    if isinstance(patch_size, int):
        patch_width, patch_height = patch_size, patch_size
    else:
        patch_width, patch_height = patch_size[0], patch_size[1]

    if stride is None:
        stride_width, stride_height = patch_width, patch_height
    else:
        if isinstance(stride, int):
            stride_width, stride_height = stride, stride
        else:
            stride_width, stride_height = stride[0], stride[1]

    y_starts_list = extract_starts(image_height, patch_height, stride_height)
    x_starts_list = extract_starts(image_width, patch_width, stride_width)

    patch_indexs = [HRImageIndex(x=x, y=y, width=patch_width, height=patch_height)
                            for y in y_starts_list
                            for x in x_starts_list]
    return patch_indexs


def NPImageGather(patch_list: List[np.ndarray],
                  index_list: List[HRImageIndex],
                  weight: np.ndarray = None) -> np.ndarray:
    r"""
      Aggregation function that combines a set of patches and their corresponding indices into a complete image.
    :param patch_list: a set of patches
    :param index_list: a set of indexes
    :param weight: If provided, its shape should match that of `patch_list`, and it will be used to aggregate the patches with specific weights.
    :return: aggregated image
    """
    assert len(patch_list) == len(index_list)
    shapes = [ patch.shape for patch in patch_list]
    assert shapes.count(shapes[0]) == len(shapes)
    ndim = patch_list[0].ndim

    if ndim ==3:
        channel = patch_list[0].shape[-1]

    image_width = max([ index.x + index.width  for index in index_list])
    image_height = max([ index.y + index.height  for index in index_list])

    if ndim == 2:
        image = np.zeros((image_height, image_width)).astype(float)
        pixel_count = np.zeros((image_height, image_width)).astype(float)
    else:
        image = np.zeros((image_height, image_width, channel)).astype(float)
        pixel_count = np.zeros((image_height, image_width, channel)).astype(float)

    for patch, index in zip(patch_list, index_list):
        x_start = index.x
        x_end = index.x + index.width

        y_start = index.y
        y_end = index.y + index.height

        if ndim == 3:
            if weight is not None:
                image[y_start: y_end, x_start: x_end, :] += patch * weight
                pixel_count[y_start:y_end, x_start: x_end, :] += weight
            else:
                image[y_start: y_end, x_start: x_end, :] += patch
                pixel_count[y_start:y_end, x_start: x_end, :] += 1
        else:
            if weight is not None:
                image[y_start: y_end, x_start: x_end] += patch * weight
                pixel_count[y_start:y_end, x_start: x_end] += weight
            else:
                image[y_start: y_end, x_start: x_end] += patch
                pixel_count[y_start:y_end, x_start: x_end] += 1

    assert np.all(pixel_count != 0)
    return image / pixel_count