import glob
import importlib

from ..util.net import Registry

LOSS = Registry('Loss')

from .base_loss import *
from .cls_loss import *
from .gan_loss import *


def get_loss_terms(loss_terms, device='cpu'):
	terms = {}
	for t in loss_terms:
		t = {k: v for k, v in t.items()}
		t_type = t.pop('type')
		t_name = t.pop('name')
		terms[t_name] = LOSS.get_module(t_type)(**t).to(device).eval()
	return terms
