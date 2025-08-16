import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as F_tv
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from . import LOSS

__all__ = ['CE', 'LabelSmoothingCE', 'SoftTargetCE']


@LOSS.register_module
class CE(nn.CrossEntropyLoss):
	def __init__(self, lam=1):
		super(CE, self).__init__()
		self.lam = lam

	def forward(self, input, target):
		return super(CE, self).forward(input, target) * self.lam


@LOSS.register_module
class LabelSmoothingCE(nn.Module):
	"""
	NLL loss with label smoothing.
	"""
	def __init__(self, smoothing=0.1, lam=1):
		"""
		Constructor for the LabelSmoothing module.
		:param smoothing: label smoothing factor
		"""
		super(LabelSmoothingCE, self).__init__()
		assert smoothing < 1.0
		self.smoothing = smoothing
		self.lam = lam
		self.confidence = 1. - smoothing

	def forward(self, x, target):
		logprobs = F.log_softmax(x, dim=-1)
		nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
		nll_loss = nll_loss.squeeze(1)
		smooth_loss = -logprobs.mean(dim=-1)
		loss = self.confidence * nll_loss + self.smoothing * smooth_loss
		return loss.mean() * self.lam


@LOSS.register_module
class SoftTargetCE(nn.Module):
	def __init__(self, lam=1, fp32=False):
		super(SoftTargetCE, self).__init__()
		self.lam = lam
		self.fp32 = fp32

	def forward(self, x, target):
		if self.fp32:
			loss = torch.sum(-target * F.log_softmax(x.float(), dim=-1), dim=-1)
		else:
			loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
		return loss.mean() * self.lam

