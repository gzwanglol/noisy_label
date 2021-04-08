ratios=(0.2 0.5 0.8 0.9)
for ratio in ${ratios[@]}; do
    python train_ce.py --r ${ratio} --data_path "./data/cifar-10-batches-py" --num_class 10 --dataset "cifar10" --gpuid $1 --exp_name "base"
done