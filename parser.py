import argparse

parser = argparse.ArgumentParser(description='Reweighted Pruning for 3D CNN')
parser.add_argument('--load-path', default='',
                    help='loading path of pretrained model')
parser.add_argument('--logger', action='store_true', default=True,
                    help='whether to use logger')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='record training status for every N batches')

parser.add_argument('--seed', type=int, default=2022, metavar='S')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers')

parser.add_argument('--arch', default='c3d',
                    choices=['c3d', 'r2+1d'])
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='total batch size of all GPUs')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--optimizer', type=str, default='sgd',
                    choices=['sgd', 'adam'])
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR',
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', type=str, default='default',
                    choices=['default', 'multistep', 'cosine'],
                    help='lr scheduler')

parser.add_argument('--config-file', type=str, nargs='+',
                    help='yaml file(s) of pruning configurations')
parser.add_argument('--sparsity', type=str, nargs='+',
                    choices=['kgr', 'kgc'])
parser.add_argument('--rew', action='store_true', default=False,
                    help='reweighted training')
parser.add_argument('--masked-retrain', action='store_true', default=False,
                    help='masked retraining after reweighted training')
parser.add_argument('--penalty', type=float, default=1e-6,
                    help='penalty parameter for reweighted pruning')

parser.add_argument('--dataset', default='ucf101',
                    choices=['ucf101'])
parser.add_argument('--group-size', nargs='*', type=int, default=[4, 4],
                    help='group size for pruning')

## tricks
parser.add_argument('--warmup', action='store_true', default=False,
                    help='warm-up scheduler')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='M',
                    help='warmup-lr, smaller than original lr')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
                    help='label smoothing, rate [0.0, 1.0], set to 0.0 to disable')

## distillation
parser.add_argument('--distill', action='store_true', default=False ,
                    help='if distillation')
parser.add_argument('--temperature', default=3, type=float,
                    help='temperature of distillation')
parser.add_argument('--kd-coefficient', default=0.5, type=float,
                    help='loss coefficient of knowledge distillation')

## mixed-precision
parser.add_argument('--amp', default=False, action='store_true',
                    help='use AMP for mixed precision training')