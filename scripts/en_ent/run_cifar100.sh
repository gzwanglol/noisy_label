ratios=(0.5 0.8 0.9)
for r in ${ratios[@]}
do
    python train_ce_ent.py \
        --r ${r} \
        --lambda_u 0 \
        --seed 0 \
        --num_class 100 \
        --gpuid $1 \
        --data_path "./data/cifar-100-python" \
        --dataset "cifar100" \
        --exp_name "ce_ent"
done

