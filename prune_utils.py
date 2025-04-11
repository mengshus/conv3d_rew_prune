import torch
import numpy as np
import yaml


def reshape_matrix2block(matrix, blk_h, blk_w):
    block = torch.cat(torch.split(matrix, blk_h), dim=1)
    block = torch.split(block, blk_w, dim=1)
    block = torch.stack([i.reshape(-1) for i in block])
    return block
def reshape_block2matrix(block, num_blk_h, num_blk_w, blk_h, blk_w):
    matrix = []
    for i in range(num_blk_h):
        for j in range(blk_h):
            matrix.append(block[num_blk_w*i:num_blk_w*(i+1), blk_w*j:blk_w*(j+1)].reshape(-1))
    matrix = torch.stack(matrix)
    return matrix


def reshape_matrix2block_kernel(matrix, blk_h, blk_w):
    block = torch.cat(torch.split(matrix, blk_h), dim=1)
    block = torch.split(block, blk_w, dim=1)
    block = torch.cat([i.permute(2, 0, 1).reshape(-1, blk_h*blk_w) for i in block])
    return block
def reshape_block2matrix_kernel(block, num_blk_h, num_blk_w, blk_h, blk_w, kernel_size):
    matrix = []
    blocks = torch.stack(torch.split(block, kernel_size), dim=1).permute(1, 2, 0)
    for i in range(num_blk_h):
        for j in range(blk_h):
            matrix.append(blocks[num_blk_w*i:num_blk_w*(i+1), blk_w*j:blk_w*(j+1)].reshape(-1))
    matrix = torch.stack(matrix)
    return matrix


def reshape_matrix2block_limit(matrix, blk_h, blk_w, mode):
    assert mode in ['row', 'col']
    block = torch.cat(torch.split(matrix, blk_h), dim=1)
    block = torch.split(block, blk_w, dim=1)
    if mode == 'row':
        return torch.stack([i.reshape(blk_h, -1) for i in block])
    else:
        return torch.stack([i.transpose(0, 1).reshape(blk_w, -1) for i in block])
        

def get_block_limit(weight_, l2_norm, mode, prune_ratio):
    assert mode in ['row', 'col']
    if mode == 'row':
        blk_limit = blk_h
    else:
        blk_limit = blk_w

    '''norm & pruning'''    
    norm, ind = torch.sort(l2_norm)
    ind_ext = []
    for i, row in enumerate(ind):
        ind_ext.append([[i, j.item()] for j in row])
    ind_ext = torch.tensor(ind_ext)
    prune_num = int(np.ceil(blk_limit * prune_ratio))
    under_threshold = ind_ext[:, :prune_num].reshape(-1, 2)
    
    for i in under_threshold:
        weight_[tuple(i)] = 0

    if mode == 'row':
        weight_ = weight_.reshape(weight_.shape[0], -1)
    else:
        weight_ = weight_.transpose(1, 2)
        weight_copy = []
        for row in weight_:
            row = torch.cat([i.transpose(0, 1) for i in torch.split(row, int(kernel_size))]).reshape(-1)
            weight_copy.append(row)
        weight_ = torch.stack(weight_copy)

    return weight_


def weight_pruning(args, c, weight_in, prune_val=None, get_norm=False):
    '''
    prune_val (float between 0-1): target sparsity of weights
    '''
    sparsity = args.sparsity[c]

    device = weight_in.get_device()  # get device (gpu), process in cpu, and put back to device
    weight = weight_in.cpu().detach()
    # weight = weight_in.clone().detach()  # gpu tensor    

    global shape, ext_shape, blk_h, blk_w, num_blk_h, num_blk_w, kernel_size
    shape = weight.shape
    block_shape = args.group_size

    ext_shape = [(shape[i] + block_shape[i] - 1) // block_shape[i] * block_shape[i] for i in range(2)] + list(shape[2:])

    blk_h, blk_w = block_shape
    num_blk_h, num_blk_w = ext_shape[0]//blk_h, ext_shape[1]//blk_w
    kernel_size = torch.prod(torch.tensor(shape[2:]))

    padding = torch.nn.ZeroPad2d((0, ext_shape[1] - shape[1], 0, ext_shape[0] - shape[0]))
    weight_ = weight.reshape(shape[0], shape[1], -1)
    weight_ = torch.stack([padding(weight_[:, :, i]) for i in range(kernel_size)], dim=2)    

    if sparsity == 'kgr':
        weight_ = reshape_matrix2block_limit(weight_, blk_h, blk_w, 'row')
        l2_norm = torch.norm(weight_, 2, dim=-1)
        if get_norm:
            return l2_norm

        weight_ = get_block_limit(weight_, l2_norm, 'row', prune_val)
        weight_ = reshape_block2matrix(weight_, num_blk_h, num_blk_w, blk_h, blk_w * kernel_size)        
        weight = weight_.reshape(ext_shape)[:shape[0], :shape[1]]

    elif sparsity == 'kgc':
        K_BUF_MAX = 9
        ksize_ext = (int((kernel_size-1)/K_BUF_MAX)+1)*K_BUF_MAX
        concat = torch.zeros(ext_shape[0], ext_shape[1], ksize_ext-kernel_size)
        weight_ = torch.cat((weight_, concat), dim=2)
        weight = torch.zeros(ext_shape)
        l2_norm_list = []
        for i in range(0, ext_shape[0], blk_h):
            for j in range(0, ext_shape[1], blk_w):
                weight_blk = weight_[i:i+blk_h, j:j+blk_w]
                weight_temp = weight_blk.reshape(blk_h, blk_w, -1, K_BUF_MAX)
                kernel_norm = torch.norm(weight_temp, 2, dim=(0, 1, 2))
                l2_norm_list.append(kernel_norm)
                if get_norm:
                    continue
                percentile = np.percentile(kernel_norm, prune_val * 100)
                under_threshold = kernel_norm <= percentile
                weight_temp[:, :, :, under_threshold] = 0
                weight[i:i+blk_h, j:j+blk_w] = weight_blk[:, :, :kernel_size].reshape(blk_h, blk_w, *shape[2:])
        l2_norm = torch.stack(l2_norm_list)
        if get_norm:
            return l2_norm

        weight = weight[:shape[0], :shape[1]]
    
    return weight.to(device)


class REW:
    '''
    Modified for combining multiple pruning schemes (row/col)
    '''
    def __init__(self, config_list):
        self.prune_cfg_list = []

        self.init(config_list)

    def init(self, config_list):
        '''
        Config file should be in .yaml format
        '''
        nl = []
        for config in config_list:
            with open(config) as f:
                raw_dict = yaml.full_load(f)
            prune_cfg = raw_dict['prune_ratios']
            self.prune_cfg_list.append(prune_cfg)
            nl += list(prune_cfg.keys())
        self.name_list = sorted(set(nl), key=nl.index)
        assert len(self.name_list) > 0, 'no layers to prune'


def hard_prune(args, REW, model):
    print('hard prune')
    masks = {}
    for (name, W) in model.named_parameters():
        if name not in REW.name_list:
            continue
        for c, prune_cfg in enumerate(REW.prune_cfg_list):
            if name not in prune_cfg:
                continue
            W_temp = weight_pruning(args, c, W, prune_cfg[name])  # replace the data field in variable with sparse model in cuda
            zero_mask = (W_temp != 0).type(torch.float32)
            if name not in masks:
                masks[name] = zero_mask
            else:
                masks[name] *= zero_mask

        W.data = W * masks[name]
        

def rew_adjust_learning_rate(optimizer, epoch, rew_milestone):
    current_lr = None
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    if epoch - 1 in rew_milestone:
        current_lr *= 1.6
    else:
        current_lr *= 0.98
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr