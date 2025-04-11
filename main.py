import torch
import torchvision

from models import c3d

from video_utils.video_utils import get_augmentor, build_dataflow
from video_utils.video_datasets import VideoDataSetLMDB
from utils import save_checkpoint, AverageMeter, accuracy, \
    CrossEntropyLossMaybeSmooth, GradualWarmupScheduler, KL
import prune_utils
from prune_utils import weight_pruning
from profile import profile_prune

from parser import parser

import numpy as np
import os
import sys
from time import time, strftime
import shutil
import logging

args = parser.parse_args()
args.datadir = os.path.join('datasets', args.dataset + '_frame')

sparsity_list = []
row, col = args.group_size
for sparsity in args.sparsity:
    sparsity_list.append(sparsity)
blk_str = '-r{}c{}'.format(row, col)
ckpt_name = '{}_{}_{}{}'.format(args.arch, args.dataset, '-'.join(sparsity_list), blk_str)
args.ckpt_dir = os.path.join('checkpoints', ckpt_name)
if args.rew and os.path.exists(args.ckpt_dir):
    i = 1
    while os.path.exists(args.ckpt_dir + '_v{}'.format(i)):
        i += 1
    os.rename(args.ckpt_dir, args.ckpt_dir + '_v{}'.format(i))
os.makedirs(args.ckpt_dir, exist_ok=True)

if args.logger:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()            
    logger.addHandler(logging.FileHandler(os.path.join(args.ckpt_dir, ckpt_name + '_{}_{}.log'.format( \
        'rew' if args.rew else 'retrain', strftime('%m%d%Y-%H%M')))))
    global print
    print = logger.info

use_cuda = torch.cuda.is_available()
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.benchmark = True  # will result in non-determinism

if args.dataset == 'ucf101':
    num_classes = 101
elif args.dataset == 'hmdb51':
    num_classes = 51

## set up model architecture
if args.arch == 'c3d':
    model = c3d.C3D(num_classes=num_classes)
elif args.arch == 'r2+1d':
    model = torchvision.models.video.r2plus1d_18(pretrained=False)
    model.fc = torch.nn.Linear(512, num_classes)

if args.load_path == '':
    if args.arch == 'c3d':
        load_path = 'checkpoints/baseline/c3d_ucf101_dense_top1-82.8.pt'
    elif args.arch == 'r2+1d':
        load_path = 'checkpoints/baseline/r2+1d_ucf101_dense_top1-94.5.pt'
    args.load_path = load_path

''' set up teacher model structure '''
## For 3D models, there is usually not larger and similar models, 
## so use the same arch as teacher arch and baseline model as teacher model
if args.distill:
    args.teacharch = args.arch
    if args.teacharch == 'c3d':
        teacher = c3d.C3D(num_classes=num_classes)
    elif args.teacharch == 'r2+1d':
        teacher = torchvision.models.video.r2plus1d_18(pretrained=False)
        teacher.fc = torch.nn.Linear(512, num_classes)
    args.teacher_path = args.load_path

if args.arch in ['c3d', 'r2+1d']:
    scale_range = [128, 128]
    crop_size = 112

dummy_input = torch.randn(1, 3, 16, crop_size, crop_size)
flops, params, _, _ = profile_prune(model, inputs=(dummy_input, ), macs=False, prune=False)

## allocate batch_size to all available GPUs
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model)
    if args.distill:
        teacher.cuda()
        teacher = torch.nn.DataParallel(teacher)

## load teacher model
if args.distill:
    teach_checkpoint = torch.load(args.teacher_path)
    try:
        teach_state_dict = teach_checkpoint['state_dict']
    except:
        teach_state_dict = teach_checkpoint
    teacher.load_state_dict(teach_state_dict)

## loss function (criterion)
'''bag of tricks setups'''
criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps)
test_criterion = torch.nn.CrossEntropyLoss()
if use_cuda:
    criterion = criterion.cuda()
    test_criterion = test_criterion.cuda()
args.smooth = args.smooth_eps > 0.0

## data loader
kwargs = {'num_workers': args.workers, 'worker_init_fn': np.random.seed(args.seed), \
          'pin_memory': True}

train_augmentor = get_augmentor(is_train=True, image_size=crop_size, threed_data=True, 
                                version='v2', scale_range=scale_range)
train_data = VideoDataSetLMDB(root_path=os.path.join(args.datadir, 'train.lmdb'), 
                              list_file=os.path.join(args.datadir, 'train.txt'),
                              num_groups=16, transform=train_augmentor, is_train=False, seperator=' ',
                              filter_video=16, max_frame=None)
train_loader = build_dataflow(dataset=train_data, is_train=True, batch_size=args.batch_size, **kwargs)

val_augmentor = get_augmentor(is_train=False, image_size=crop_size, threed_data=True)
val_data = VideoDataSetLMDB(root_path=os.path.join(args.datadir, 'val.lmdb'), 
                            list_file=os.path.join(args.datadir, 'val.txt'),
                            num_groups=16, transform=val_augmentor, is_train=False, seperator=' ',
                            filter_video=16, max_frame=None)
val_loader = build_dataflow(dataset=val_data, is_train=False, batch_size=args.batch_size, **kwargs)

## optimizer and scheduler
optimizer_init_lr = args.warmup_lr if args.warmup else args.lr

optimizer = None
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr, momentum=0.9, weight_decay=5e-4)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)

scheduler = None
if args.lr_scheduler == 'default':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(args.epochs * 0.2)) * len(train_loader), gamma=0.5)
elif args.lr_scheduler == 'multistep':
    epoch_milestones = [45, 90, 120]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i * len(train_loader) for i in epoch_milestones], gamma=0.5)
elif args.lr_scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=4e-08)
else:
    exit('unknown lr scheduler')

if args.warmup:
    scheduler = GradualWarmupScheduler(optimizer, multiplier=args.lr / args.warmup_lr, 
                                       total_iter=args.warmup_epochs * len(train_loader), after_scheduler=scheduler)

print(' '.join(sys.argv))
print(model)
print('params: {:.4g}M   flops: {:.4g}G'.format(params/1e6, flops/1e9))
print('General config:')
for k, v in sorted(vars(args).items()):
    print('\t{}: {}'.format(k, v))


def main():
    ## reweighted train or masked retrain
    global REW
    REW = prune_utils.REW(config_list=[os.path.join('prune_settings', c + '.yaml') for c in args.config_file])
    
    print('Prune config:')
    for c, prune_cfg in enumerate(REW.prune_cfg_list):
        for k, v in prune_cfg.items():
            print('\t{}: {}'.format(k, v))
        print('')
    
    if args.rew and args.masked_retrain:
        exit('Reweighted training and masked retraining cannot be performed simultenously')
    elif args.rew:
        reweighted_training(args, model, train_loader, val_loader, criterion, optimizer, None)
    elif args.masked_retrain:
        masked_retrain(args, model, train_loader, val_loader, criterion, optimizer, scheduler)


def reweighted_training(args, model, train_loader, val_loader, criterion, optimizer, scheduler):
    for file in args.config_file:
        shutil.copy(os.path.join('prune_settings', file + '.yaml'), \
            os.path.join(args.ckpt_dir, file + '.yaml'))
        
    print('>_ loading model from {}\n'.format(args.load_path))
    checkpoint = torch.load(args.load_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    global num_config
    num_config = len(args.config_file)
    global rew_layers
    rew_layers = [{} for _ in range(num_config)]
    global eps
    eps = 1e-6
    for (name, W) in model.named_parameters():
        if name not in REW.name_list:
            continue
        for c, prune_cfg in enumerate(REW.prune_cfg_list):
            if name not in prune_cfg:
                continue
            norm = weight_pruning(args, c, W.data, get_norm=True)
            rew_layers[c][name] = 1 / (norm**2 + eps)

    global rew_milestone
    rew_milestone = [10, 20, 30, 40, 50, 75, 100, 125]  # epoch of reweighted updates

    save_path = os.path.join(args.ckpt_dir, ckpt_name + '_rew.pt')

    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, model, criterion, optimizer, scheduler, epoch)
        top1 = validate(args, model, val_loader)
        print('')

        save_checkpoint(model.state_dict(), False, save_path)


def masked_retrain(args, model, train_loader, val_loader, criterion, optimizer, scheduler):
    load_path = os.path.join(args.ckpt_dir, ckpt_name + '_rew.pt')
    print('>_ loading model from {}\n'.format(load_path))
    model.load_state_dict(torch.load(load_path))

    prune_utils.hard_prune(args, REW, model)

    top1_list = [0.]
    best_epoch = 1

    save_path = os.path.join(args.ckpt_dir, ckpt_name + '_retrain.pt')

    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, model, criterion, optimizer, scheduler, epoch)
        top1 = validate(args, model, val_loader)

        best_top1 = max(top1_list)
        is_best = top1 > best_top1     
        save_checkpoint(model.state_dict(), is_best, save_path)
        top1_list.append(top1)
        if is_best:
            best_top1 = top1
            best_epoch = epoch        
        print('Best Acc@1: {:.3f}%  Best epoch: {}\n'.format(best_top1, best_epoch))

    os.rename(save_path.replace('.pt', '_best.pt'), \
        save_path.replace('.pt', '_top1-{:.3f}.pt'.format(best_top1)))


def train(args, train_loader, model, criterion, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ce_losses = AverageMeter()
    rew_losses = AverageMeter()
    old_ce_losses = AverageMeter() if args.distill else None
    distill_losses = AverageMeter() if args.distill else None
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.rew:
        print('reweighted training')
    elif args.masked_retrain:
        print('accuracy retrain masking')
        masks = {}
        for (name, W) in model.named_parameters():
            if name not in REW.name_list:
                continue            
            weight = W.detach()
            masks[name] = (weight != 0).type(torch.float32)

    ## mixed-precision training, creates once at the beginning
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    model.train()

    epoch_start_time = time()
    end = time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time() - end)

        if args.masked_retrain:
            scheduler.step()

        if use_cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        ## cast operations to mixed precision
        if args.amp:
            with torch.cuda.amp.autocast():
                output = model(input)
        else:
            output = model(input)

        ce_loss = criterion(output, target, smooth=args.smooth)

        ## compute teacher output
        if args.distill and epoch <= args.epochs - 5:
            with torch.no_grad():
                teacher_output = teacher(input)
            distill_loss = KL(output, teacher_output, args.temperature)
            old_ce_loss = ce_loss.clone()
            ce_loss = ce_loss * (1 - args.kd_coefficient) + distill_loss * args.kd_coefficient

        if args.rew:
            if epoch > 1 and i == 0:
                prune_utils.rew_adjust_learning_rate(optimizer, epoch, rew_milestone)

            ## reweighted parameter update            
            if epoch - 1 in rew_milestone and i == 0:
                print('reweighted parameter update')
                for (name, W) in model.named_parameters():
                    if name not in REW.name_list:
                        continue
                    for c, prune_cfg in enumerate(REW.prune_cfg_list):
                        if name not in prune_cfg:
                            continue                        
                        norm = weight_pruning(args, c, W.data, get_norm=True)
                        rew_layers[c][name] = 1 / (norm**2 + eps)

            ## reweighted loss
            rew_loss = 0
            for (name, W) in model.named_parameters():
                if name not in REW.name_list:
                    continue
                for c, prune_cfg in enumerate(REW.prune_cfg_list):
                    if name not in prune_cfg:
                        continue                    
                    norm = weight_pruning(args, c, W.data, get_norm=True)
                    rew_loss += args.penalty / num_config * torch.sum(rew_layers[c][name] * norm)
            loss = ce_loss + rew_loss
            rew_losses.update(rew_loss, input.size(0))
        else:
            loss = ce_loss

        top1_list, top5_list = accuracy(output, target, topk=(1, 5))
        ce_losses.update(ce_loss.item(), input.size(0))
        top1.update(top1_list[0], input.size(0))
        top5.update(top5_list[0], input.size(0))
        if args.distill and epoch <= args.epochs - 5:
            distill_losses.update(distill_loss.item(), input.size(0))
            old_ce_losses.update(old_ce_loss.item(), input.size(0))

        
        ## update weights every 4 iterations, effective batch size x 4
        ## compute gradient
        loss /= 4
        if args.amp:
            ## scale loss, and call backward() to create scaled gradients
            scaler.scale(loss).backward()                
        else:
            loss.backward()

        ## SGD step
        if (i+1) % 4 == 0 or (i+1) == len(train_loader):
            if args.masked_retrain:
                with torch.no_grad():
                    for name, W in (model.named_parameters()):
                        if name in masks:
                            W.grad *= masks[name]

            if args.amp:
                ## unscale gradients, call or skips optimizer.step()
                scaler.step(optimizer)
                ## update scale for next iteration
                scaler.update()
            else:
                optimizer.step()

            ## reset gradients to None; PyTorch >= 1.7
            optimizer.zero_grad(set_to_none=True)

        batch_time.update(time() - end)
        end = time()

        if i % args.log_interval == 0:
            current_lr = list(optimizer.param_groups)[0]['lr']
            rew_str = '    Rew Loss {rew_loss.val:.4f} ({rew_loss.avg:.4f})   '.format( \
                rew_loss=rew_losses) if args.rew else ''
            distill_str = '    Distill Loss {distill_loss.avg:.4f}   Old CE Loss {ce_loss.avg:.4f}   '.format( \
                distill_loss=distill_losses, ce_loss=old_ce_losses) \
                if args.distill and epoch <= args.epochs - 5 else ''
            print('Epoch [{0}][{1:3d}/{2}] [({3}) lr={4:f}]   '
                  'Stage [{5}]   '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                  'CE Loss {loss.val:.4f} ({loss.avg:.4f}){rew_str}{distill_str}'
                  'Acc@1 {top1.val:7.3f} ({top1.avg:7.3f})   '
                  'Acc@5 {top5.val:7.3f} ({top5.avg:7.3f})'.format(
                epoch, i, len(train_loader), args.optimizer, current_lr, 'rew' if args.rew else 'retrain',
                batch_time=batch_time, loss=ce_losses, rew_str=rew_str, distill_str=distill_str, top1=top1, top5=top5))
    print('Train Acc@1 {:.3f}%   Time {}'.format( \
        top1.avg, int(time() - epoch_start_time)))


def validate(args, model, val_loader):  
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        epoch_start_time = time()
        end = time()
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            output = model(input)
            loss = test_criterion(output, target)

            top1_list, top5_list = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(top1_list[0], input.size(0))
            top5.update(top5_list[0], input.size(0))

            batch_time.update(time() - end)
            end = time()

            if i % args.log_interval == 0:
                print('Test [{0:3d}/{1}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Acc@1 {top1.val:7.3f} ({top1.avg:7.3f})   '
                      'Acc@5 {top5.val:7.3f} ({top5.avg:7.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print('Test Acc@1 {:.3f}%   Time {}'.format( \
            top1.avg, int(time() - epoch_start_time)))  

    return top1.avg


if __name__ == '__main__':
    main()
