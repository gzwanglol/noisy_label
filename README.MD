# Noisy label

## Dataset
We use CIFAR-10 and CIFAR-100.
Download link:

CIFAR10:
https://drive.google.com/file/d/1Mbt4tac_T3eyk2pwAIi1Qs2V4Bl0EZpN/view?usp=sharing

CIFAR100:
https://drive.google.com/file/d/1Ku_2_S3aJW7eQqQ8bi25vK_bokK4_L9l/view?usp=sharing

## Run
**.py** contains implementation of corresponding method, to run them with prefixed hyper-parameters, use **.sh** files in scripts.
|Method|.py|sh folder|
| --- | --- | --- |
|Cross-Entropy|train_ce.py|ce|
|Cross-Entropy + Entropy |train_ce_ent.py|ce_ent|
|DivideMix - GMM (random sample selection)|train_randommix.py|random_mix|
|DivideMix|Train_cifar.py|base|


e.g., to run DivideMix on CIFAR-10 with 80% of noise, use the following command
``
./scripts/base/cifar10_0.8.sh 0
``
where 0 is the GPU id.

## TODO
Divide = DivideMix - GMM

The following requires GPU to run
|Item | Comments|
| --- | --- |
|Divide + HMM| |
|~~Divide + CRF (Maybe)~~|Supervised model, not doable| 
|~~Divide + CRF + LSTM (Maybe)~~|supervised model , not doable|
|CE + Knowledge Graph (Maybe)| |
|CE + Negative Labeling + Knowledge Graph (Maybe)||

Don't have to run on a GPU.
|TODO|Tool|
| -- | -- |
|Visualize loss curve of different methods|matplotlib|
|Visualize features of each method|scikit-learn manifold tsne|
|Visualize some samples (Maybe)||
