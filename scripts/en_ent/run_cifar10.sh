ratios=(0.8 0.9)
for r in ${ratios[@]}
do
    python train_ce_ent.py \
        --r ${r} \
        --lambda_u 0 \
        --seed 0 \
        --num_class 10 \
        --gpuid $1 \
        --data_path "./data/cifar-10-batches-py" \
        --dataset "cifar10" \
        --exp_name "ce_ent"
done