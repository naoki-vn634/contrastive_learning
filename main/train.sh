# !/bin/sh
source /etc/profile.d/modules.sh
source activate meta

 python -u main.py \
--train_mode 0 \
--input '/mnt/aoni02/matsunaga/Contrastive/early_stage/pre_2048/' \
--epoch 101 \
--batchsize 32 \
--tfboard True \
--output '/mnt/aoni02/matsunaga/Contrastive/early_stage/pre_2048_crop' \
--optim Adam \
--crop_rate 0.5

python -u main.py \
--train_mode 1 \
--input "/mnt/aoni02/matsunaga/Contrastive/early_stage/pre_2048/" \
--epoch 501 \
--batchsize 32 \
--tfboard True \
--output "/mnt/aoni02/matsunaga/Contrastive/early_stage/pre_finetune_2048_crop" \
--optim Adam \
--crop_rate 0.5 \
--cls_head_one True \
--checkpoint_iter 50 \
--weight "/mnt/aoni02/matsunaga/Contrastive/early_stage/pre_2048_crop/weight/99_weight.pth"
