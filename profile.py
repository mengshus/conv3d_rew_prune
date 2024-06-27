# modified from thop.profile

import logging

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


### count_hooks
def zero_ops(m, x, y):
    m.total_ops += torch.Tensor([int(0)])


def count_convNd(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    kernel_ops = m.weight.size()[2:].numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    # total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops + bias_ops)
    assert m.in_channels // m.groups * kernel_ops >= 1
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops * 2 - 1 + bias_ops)

    m.total_ops += torch.Tensor([int(total_ops)])
    m.out_size = torch.Tensor([int(y.nelement())])


def count_convNd_ver2(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    # N x H x W (exclude Cout)
    output_size = (y.size()[:1] + y.size()[2:]).numel()
    # Cout x Cin x Kw x Kh
    kernel_ops = m.weight.nelement()
    if m.bias is not None:
        # Cout x 1
        kernel_ops += + m.bias.nelement()
    # x N x H x W x Cout x (Cin x Kw x Kh + bias)
    m.total_ops += torch.Tensor([int(output_size * kernel_ops)])


def count_bn(m, x, y):
    x = x[0]

    nelements = x.numel()
    # subtract, divide, gamma, beta
    total_ops = 4 * nelements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()

    m.total_ops += torch.Tensor([int(nelements)])


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size]))
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_adap_avgpool(m, x, y):
    kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


# TODO: verify the accuracy
def count_upsample(m, x, y):
    if m.mode not in ("nearest", "linear", "bilinear", "bicubic",):  # "trilinear"
        logger.warning("mode %s is not implemented yet, take it a zero op" % m.mode)
        return zero_ops(m, x, y)

    if m.mode == "nearest":
        return zero_ops(m, x, y)

    x = x[0]
    if m.mode == "linear":
        total_ops = y.nelement() * 5  # 2 muls + 3 add
    elif m.mode == "bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = y.nelement() * 11  # 6 muls + 5 adds
    elif m.mode == "bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] x [4x4] x [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops = y.nelement() * (ops_solve_A + ops_solve_p)
    elif m.mode == "trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        total_ops = y.nelement() * (13 * 2 + 5)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])
    m.out_size = torch.Tensor([int(y.numel())])
###

register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,

    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,

    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops,

    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample
}


def get_state_dict(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    try:
        state_dict = checkpoint['state_dict']
    except:
        state_dict = checkpoint
    new_state_dict = {}
    for key, value in state_dict.items():
        key = key.split('module.')[-1]
        new_state_dict[key] = value
    return new_state_dict

def get_prune_cfg(cfg_file):
    import yaml
    with open(cfg_file, 'r') as stream:
        raw_dict = yaml.full_load(stream)
    if 'prune_ratios' in raw_dict:
        prune_cfg = raw_dict['prune_ratios']
    new_prune_cfg = {}
    for key, value in prune_cfg.items():
        key = key.split('module.')[-1]
        new_prune_cfg[key] = value
    return new_prune_cfg

def profile_prune(model, inputs, macs=False, custom_ops=None, verbose=False, \
    prune=True, mode=0, file=None, show_layer=True, inMega=False):
    '''
    mode = 0: file is path of pruned model (.pt, .pth, etc.)
    mode = 1: file is path of pruning configuration file (.yaml)
    '''

    handler_collection = []
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        if hasattr(m, "total_ops") or hasattr(m, "total_params"):
            logger.warning("Either .total_ops or .total_params is already defined in %s."
                           "Be careful, it might change your code's behavior." % str(m))

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))
        m.register_buffer('out_size', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is None:
            if verbose:
                print("THOP has not implemented counting method for ", m)
        else:
            if verbose:
                print("Register FLOP counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    op_name = 'macs' if macs else 'ops'
    op_num = 1e6 if inMega else 1e9
    op_unit = 'M' if inMega else 'G'

    ops = {}
    params = {}
    fsize = {}

    # for m in model.modules():
    #     if len(list(m.children())) > 0:  # skip for non-leaf module
    #         continue
    #     total_ops += m.total_ops
    #     total_params += m.total_params
    for name, m in model.named_modules():  # layer name (...conv, fc, etc.)
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue

        ops[name] = m.total_ops.item()
        if macs: ops[name] /= 2
        params[name] = m.total_params.item()
        fsize[name] = m.out_size.item()
        if params[name] == 0:
            continue
        
        if not prune and show_layer and (isinstance(m, nn.modules.conv._ConvNd) or isinstance(m, nn.Linear)):  # conv or fc
            print(name)
            print('   params: {:.2f}M   ofm size: {:<6.0f}   {}: {:.2f}{}'.format(
                params[name]/1e6, fsize[name], op_name, ops[name]/op_num, op_unit))
            print('')

    total_ops = sum(ops.values())
    total_params = sum(params.values())

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    # remove temporal buffers
    for n, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if "total_ops" in m._buffers:
            m._buffers.pop("total_ops")
        if "total_params" in m._buffers:
            m._buffers.pop("total_params")
        if "out_size" in m._buffers:
            m._buffers.pop("out_size")

    if not prune:
        print('total:\n   params: {:.2f}M   max ofm size: {:<6.0f}   '
            'total {}: {:.2f}{}'.format(total_params/1e6, max(fsize.values()), op_name, total_ops/op_num, op_unit))
        return total_ops, total_params, None, None

    #################### prune ####################
    if mode == 0:
        state_dict = get_state_dict(file)
    else:
        prune_cfg = get_prune_cfg(file)

    params_rmn = {}
    ops_rmn = {}

    for name, m in model.named_modules():  # layer name (...conv, fc, etc.)
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue

        if params[name] == 0:
            continue

        num = 0
        for p_name, p in m.named_parameters():  # parameter name (weight, bias, etc.)
            key = '.'.join([name, p_name])
            if mode == 0:
                num += len((state_dict[key] != 0).nonzero())
            else:                     
                if key in prune_cfg:
                    assert 0 <= prune_cfg[key] <= 1
                    num += p.numel() * (1 - prune_cfg[key])
                else:
                    num += p.numel()
        params_rmn[name] = num
        ops_rmn[name] = float(ops[name]) * params_rmn[name] / params[name]

        if show_layer and (isinstance(m, nn.modules.conv._ConvNd) or isinstance(m, nn.Linear)):  # conv or fc
            print(name)
            print('   params rmn/tot: {:.2f}M/{:.2f}M ({:.2f}x, {:.2f} pruned)   ofm size: {:<6.0f}   '
                '{} rmn/tot: {:.2f}{}/{:.2f}{}'.format(
                params_rmn[name]/1e6, params[name]/1e6, params[name]/params_rmn[name], 1-params_rmn[name]/params[name], \
                fsize[name], op_name, ops_rmn[name]/op_num, op_unit, ops[name]/op_num, op_unit))
            print('')

    total_params_rmn = sum(params_rmn.values())
    total_ops_rmn = sum(ops_rmn.values())

    print('total:\n   params rmn/tot: {:.2f}M/{:.2f}M ({:.2f}x)   max ofm size: {:<6.0f}   '
        '{} rmn/tot: {:.2f}{}/{:.2f}{} ({:.2f}x)'.
        format(total_params_rmn/1e6, total_params/1e6, total_params/total_params_rmn, \
            max(fsize.values()), op_name, total_ops_rmn/op_num, op_unit, total_ops/op_num, op_unit, total_ops/total_ops_rmn))

    return total_ops, total_params, total_ops_rmn, total_params_rmn