from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F


class VitADConfig(Namespace):
	def __init__(self,
                 backbone_name,
                 image_size):

		self.epoch_full = 200
		self.warmup_epochs = 0
		self.eval_per_steps = 100
		self.start_eval_steps = 200
		self.log_per_steps = 10

		self.lr = 1e-4
		self.weight_decay = 0.0001

		self.model_t = Namespace()
		self.model_t.name = backbone_name
		self.model_t.kwargs = dict(pretrained=True,
								   pretrained_strict = False,
                                   img_size=image_size,
								   teachers=[3, 6, 9],
								   neck=[12])

		self.model_f = Namespace()
		self.model_f.name = 'fusion'
		self.model_f.kwargs = dict(pretrained=False, dim=768, mul=1)

		self.model_s = Namespace()
		self.model_s.name = f'de_{backbone_name}'
		self.model_s.kwargs = dict(pretrained=False,
								   img_size=image_size, students=[3, 6, 9], depth=9)

		self.model = Namespace()
		self.model.name = 'vitad'
		self.model.kwargs = dict(pretrained=False, model_t=self.model_t,
								 model_f=self.model_f, model_s=self.model_s)

		# ==> optimizer
		self.optim = Namespace()
		self.optim.kwargs = dict(name='adamw', betas=(0.9, 0.999), eps=1e-8, weight_decay=self.weight_decay, amsgrad=False)
		self.optim.lr = self.lr

		# ==> trainer
		self.trainer = Namespace()
		self.trainer.name = 'ViTADTrainer'
		self.trainer.epoch_full = self.epoch_full

		self.trainer.scheduler_kwargs = dict(
			name='step', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=self.lr / 1e2,
			warmup_lr=self.lr / 1e3, warmup_iters=-1, cooldown_iters=0, warmup_epochs=self.warmup_epochs, cooldown_epochs=0, use_iters=True,
			patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=int(self.epoch_full * 0.8), cycle_decay=0.1, decay_rate=0.1)

		self.trainer.mixup_kwargs = dict(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=0.0, switch_prob=0.5, mode='batch', correct_lam=True, label_smoothing=0.1)
		self.trainer.eval_per_steps = self.eval_per_steps
		self.trainer.start_eval_steps = self.start_eval_steps

		self.loss = Namespace()
		self.loss.clip_grad = 5.0
		self.loss.loss_terms = [
			dict(type='CosLoss', name='cos', avg=False, lam=1.0),
		]


