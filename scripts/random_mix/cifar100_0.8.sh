python train_randommix.py \
    --r 0.8 \
    --lambda_u 150 \
    --seed 0 \
    --num_class 100 \
    --gpuid $1 \
    --data_path "./data/cifar-100-python" \
    --dataset "cifar100" \
    --exp_name "random_mix"