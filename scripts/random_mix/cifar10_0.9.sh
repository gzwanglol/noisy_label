python train_randommix.py \
    --r 0.9 \
    --lambda_u 50 \
    --seed 0 \
    --num_class 10 \
    --gpuid $1 \
    --data_path "./data/cifar-10-batches-py" \
    --dataset "cifar10" \
    --exp_name "random_mix"