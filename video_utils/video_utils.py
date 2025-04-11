import numpy as np
import shutil
import os
import subprocess
import time
import multiprocessing

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from .video_transforms import (GroupRandomHorizontalFlip,
                               GroupMultiScaleCrop, GroupScale, GroupCenterCrop, GroupRandomCrop,
                               GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomScale)


def map_charades(y_pred, y_true):
    """ Returns mAP """
    y_true = y_true.cpu().numpy().astype(np.int32)
    y_pred = y_pred.cpu().detach().numpy()
    m_aps = []
    n_classes = y_pred.shape[1]
    for oc_i in range(n_classes):
        pred_row = y_pred[:, oc_i]
        sorted_idxs = np.argsort(-pred_row)
        true_row = y_true[:, oc_i]
        tp = true_row[sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(np.nan)
            continue
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs + t_pcs).astype(float)
        avg_prec = 0
        for i in range(y_pred.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    m_ap = m_ap if not np.isnan(m_ap) else 0.0
    return [m_ap], [0]


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.decode().strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def get_augmentor(is_train, image_size, mean=None,
                  std=None, disable_scaleup=False, is_flow=False,
                  threed_data=False, version='v1', scale_range=None):

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    scale_range = [256, 320] if scale_range is None else scale_range
    augments = []

    if is_train:
        if version == 'v1':
            augments += [
                GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
            ]
        elif version == 'v2':
            augments += [
                GroupRandomScale(scale_range),
                GroupRandomCrop(image_size),
            ]
        augments += [GroupRandomHorizontalFlip(is_flow=is_flow)]
    else:
        scaled_size = image_size if disable_scaleup else int(image_size / 0.875 + 0.5)
        augments += [
            GroupScale(scaled_size),
            GroupCenterCrop(image_size)
        ]
    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor


def my_collate(batch):
        batch = filter(lambda x:x is not None, batch)
        return torch.utils.data.dataloader.default_collate(list(batch))


def build_dataflow(dataset, is_train, batch_size, is_distributed=False, **kwargs):
    # workers = min(workers, multiprocessing.cpu_count())
    shuffle = False

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    if is_train:
        shuffle = sampler is None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              sampler=sampler, collate_fn=my_collate, **kwargs)

    return data_loader


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    result = torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
    return result


# additional subgradient descent on the sparsity-induced penalty term
def updateBN(model, c):
    l1_loss = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            # print('found bn layer: {}'.format(m))
            m.weight.grad.data.add_(c*torch.sign(m.weight.data))  # L1
            l1_loss += m.weight.abs().sum()
    return l1_loss


def mask_bn(model, bn_masks):
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.mul_(bn_masks[k])
            m.bias.data.mul_(bn_masks[k])

def mask_bn_grad(model, bn_masks):
    with torch.no_grad():
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.mul_(bn_masks[k])
                m.bias.grad.data.mul_(bn_masks[k])

def mask_conv(model, bn_masks):
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) and isinstance(list(model.modules())[k+1], nn.BatchNorm2d):
            bn_mask = bn_masks[k+1]
            shape = m.weight.data.shape
            conv_mask = torch.zeros(shape)
            conv_mask_2d = conv_mask.reshape(shape[0], -1)
            for i in range(conv_mask_2d.shape[1]):
                conv_mask_2d[:, i] = bn_mask
            if m.weight.device.type == ('cuda'):
                conv_mask = conv_mask.cuda()
            m.weight.data.mul_(conv_mask)
            
    masks = {}
    for name, P in model.named_parameters():
        param = P.detach()
        non_zeros = param != 0
        masks[name] = non_zeros.type(torch.float32)
    return masks

