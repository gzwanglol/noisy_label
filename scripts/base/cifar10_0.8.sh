python Train_cifar.py \
    --r 0.8 \
    --lambda_u 25 \
    --seed 0 \
    --num_class 10 \
    --gpuid $1 \
    --data_path "./data/cifar-10-batches-py" \
    --dataset "cifar10" 