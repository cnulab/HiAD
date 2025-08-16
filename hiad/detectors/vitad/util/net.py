import torch
import random
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


class Registry:

	def __init__(self, name):
		self.name = name
		self.name_to_fn = dict()

	def register_module(self, fn, name=None):
		module_name = name if name else fn.__name__
		self.name_to_fn[module_name] = fn
		return fn

	def __len__(self):
		return len(self.name_to_fn)

	def __contains__(self, name):
		return name in self.name_to_fn.keys()

	def get_module(self, name):
		if self.__contains__(name):
			return self.name_to_fn[name]
		else:
			raise ValueError('invalid module: {}'.format(name))


def cal_anomaly_map(ft_list, fs_list, out_size, uni_am=False, use_cos=True, amap_mode='add',
					gaussian_sigma=0, weights=None):

	bs = ft_list[0].shape[0]
	weights = weights if weights else [1] * len(ft_list)
	anomaly_map = np.ones([bs, out_size[1], out_size[0]]) if amap_mode == 'mul' else np.zeros([bs, out_size[1], out_size[0]])
	a_map_list = []
	if uni_am:
		size = (ft_list[0].shape[2], ft_list[0].shape[3])
		for i in range(len(ft_list)):
			ft_list[i] = F.interpolate(F.normalize(ft_list[i], p=2), size=size, mode='bilinear', align_corners=True)
			fs_list[i] = F.interpolate(F.normalize(fs_list[i], p=2), size=size, mode='bilinear', align_corners=True)
		ft_map, fs_map = torch.cat(ft_list, dim=1), torch.cat(fs_list, dim=1)
		if use_cos:
			a_map = 1 - F.cosine_similarity(ft_map, fs_map, dim=1)
			a_map = a_map.unsqueeze(dim=1)
		else:
			a_map = torch.sqrt(torch.sum((ft_map - fs_map) ** 2, dim=1, keepdim=True))
		a_map = F.interpolate(a_map, size=(out_size[1], out_size[0]), mode='bilinear', align_corners=True)
		a_map = a_map.squeeze(dim=1).cpu().detach().numpy()
		anomaly_map = a_map
		a_map_list.append(a_map)
	else:
		for i in range(len(ft_list)):
			ft = ft_list[i]
			fs = fs_list[i]

			if use_cos:
				a_map = 1 - F.cosine_similarity(ft, fs, dim=1)
				a_map = a_map.unsqueeze(dim=1)
			else:
				a_map = torch.sqrt(torch.sum((ft - fs) ** 2, dim=1, keepdim=True))
			a_map = F.interpolate(a_map, size=(out_size[1], out_size[0]), mode='bilinear', align_corners=True)
			a_map = a_map.squeeze(dim=1)
			a_map = a_map.cpu().detach().numpy()
			a_map_list.append(a_map)
			if amap_mode == 'add':
				anomaly_map += a_map * weights[i]
			else:
				anomaly_map *= a_map
		if amap_mode == 'add':
			anomaly_map /= (len(ft_list) * sum(weights))
	if gaussian_sigma > 0:
		for idx in range(anomaly_map.shape[0]):
			anomaly_map[idx] = gaussian_filter(anomaly_map[idx], sigma=gaussian_sigma)
	return anomaly_map, a_map_list



def set_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)