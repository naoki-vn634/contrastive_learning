# !/bin/sh
source /etc/profile.d/modules.sh
source activate meta

 python -u main.py \
--train_mode 0 \
--hidden 1 \
--input '/mnt/aoni02/matsunaga/Dataset/200313_global-model-ranch/garbage_ver3/train' \
--epoch 101 \
--batchsize 32 \
--tfboard True \
--output '/mnt/aoni02/matsunaga/Contrastive/each_ranch/pre_2048' \
--optim Adam \
--crop_rate 0.5

python -u main.py \
--train_mode 1 \
-- hidden 1 \
--input "/mnt/aoni02/matsunaga/Contrastive/each_ranch/pre_2048" \
--epoch 501 \
--batchsize 32 \
--tfboard True \
--output "/mnt/aoni02/matsunaga/Contrastive/each_ranch/pre_finetune_2048" \
--optim Adam \
--crop_rate 0.5 \
--cls_head_one True \
--checkpoint_iter 50 \
--weight "/mnt/aoni02/matsunaga/Contrastive/each_ranch/pre_2048/weight/100_weight.pth"
