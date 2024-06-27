CUDA_VISIBLE_DEVICES=0 python main.py --log-interval 20 \
  --arch c3d --dataset ucf101 \
  --rew --sparsity kgr kgc --group-size 8 8 \
  --config-file c3d_1.76x_uniform c3d_1.93x \
  --epoch 150 --batch-size 32 --lr 2e-4 --optimizer sgd \
  --smooth-eps 0.1 &&
CUDA_VISIBLE_DEVICES=0 python main.py --log-interval 20 \
  --arch c3d --dataset ucf101 \
  --masked-retrain --sparsity kgr kgc --group-size 8 8 \
  --config-file c3d_1.76x_uniform c3d_1.93x \
  --epoch 90 --batch-size 32 --lr 5e-4 --optimizer sgd \
  --lr-scheduler cosine \
  --warmup --warmup-lr 1e-5 \
  --distill