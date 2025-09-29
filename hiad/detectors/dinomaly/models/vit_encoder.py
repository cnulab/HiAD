import torch
import timm
from torch.hub import HASH_REGEX, download_url_to_file, urlparse
import numpy as np
from scipy import interpolate
import os

from ..dinov1 import vision_transformer
from ..dinov2.models import vision_transformer as vision_transformer_dinov2

_WEIGHTS_DIR = "backbones/weights"
os.makedirs(_WEIGHTS_DIR, exist_ok=True)

def load(name):

    arch, patchsize = name.split("_")[-2], name.split("_")[-1]
    model = vision_transformer.__dict__[f'vit_{arch}'](patch_size=int(patchsize))
    if "dino" in name:
        if "v2" in name:
            if "reg" in name:
                model = vision_transformer_dinov2.__dict__[f'vit_{arch}'](patch_size=int(patchsize), img_size=518,
                                                                          block_chunks=0, init_values=1e-8,
                                                                          num_register_tokens=4,
                                                                          interpolate_antialias=False,
                                                                          interpolate_offset=0.1)

                if arch == "base":
                    ckpt_pth = download_cached_file(
                        f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb{patchsize}/dinov2_vitb{patchsize}_reg4_pretrain.pth")
                elif arch == "small":
                    ckpt_pth = download_cached_file(
                        f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vits{patchsize}/dinov2_vits{patchsize}_reg4_pretrain.pth")
                elif arch == "large":
                    ckpt_pth = download_cached_file(
                        f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl{patchsize}/dinov2_vitl{patchsize}_reg4_pretrain.pth")
                else:
                    raise ValueError("Invalid type of architecture. It must be either 'small' or 'base' or 'large.")
            else:
                model = vision_transformer_dinov2.__dict__[f'vit_{arch}'](patch_size=int(patchsize), img_size=518,
                                                                          block_chunks=0, init_values=1e-8,
                                                                          interpolate_antialias=False,
                                                                          interpolate_offset=0.1)

                if arch == "base":
                    ckpt_pth = download_cached_file(
                        f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb{patchsize}/dinov2_vitb{patchsize}_pretrain.pth")
                elif arch == "small":
                    ckpt_pth = download_cached_file(
                        f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vits{patchsize}/dinov2_vits{patchsize}_pretrain.pth")
                else:
                    raise ValueError("Invalid type of architecture. It must be either 'small' or 'base'.")

            state_dict = torch.load(ckpt_pth, map_location='cpu')
        else:  # dinov1
            if arch == "base":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dino/dino_vit{arch}{patchsize}_pretrain/dino_vit{arch}{patchsize}_pretrain.pth")
            elif arch == "small":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dino/dino_deit{arch}{patchsize}_pretrain/dino_deit{arch}{patchsize}_pretrain.pth")
            else:
                raise ValueError("Invalid type of architecture. It must be either 'small' or 'base'.")

            state_dict = torch.load(ckpt_pth, map_location='cpu')

    model.load_state_dict(state_dict, strict=False)
    return model


def download_cached_file(url, check_hash=True, progress=True):
    """
    Mostly copy-paste from timm library.
    (https://github.com/rwightman/pytorch-image-models/blob/29fda20e6d428bf636090ab207bbcf60617570ca/timm/models/_hub.py#L54)
    """
    if isinstance(url, (list, tuple)):
        url, filename = url
    else:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
    cached_file = os.path.join(_WEIGHTS_DIR, filename)
    if not os.path.exists(cached_file):
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file


def convert_key(ckpt_pth):
    ckpt = torch.load(ckpt_pth, map_location="cpu")
    state_dict = ckpt['state_dict']
    new_state_dict = dict()

    for k, v in state_dict.items():
        if k.startswith('module.base_encoder.'):
            new_state_dict[k[len("module.base_encoder."):]] = v

    return new_state_dict


def beit_checkpoint_process(checkpoint_model, model):
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)
        if "head." in key:
            checkpoint_model.pop(key)
        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.grid_size
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                print("Position interpolate for %s from %dx%d to %dx%d" % (
                    key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                # print("Original positions = %s" % str(x))
                # print("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias
    # interpolate position embedding
    if ('pos_embed' in checkpoint_model) and (model.pos_embed is not None):
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            # print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
