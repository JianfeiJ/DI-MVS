#!/usr/bin/env bash
MVS_TRAINING=""
BLEND_TRAINING=""
LOG_DIR="checkpoints"
LOG_DIR_CKPT="checkpoints"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi
filename=""
dirAndName="$LOG_DIR/$filename.log"
if [ ! -d $dirAndName ]; then
    touch $dirAndName
fi

##DTU
python -u train.py --mode='train' --epochs=16 --numdepth=384 --trainviews=5 --lr=0.001 --testviews=5 --logdir $LOG_DIR --dataset=dtu_yao_1to8_inverse --batch_size=6 --trainpath=$MVS_TRAINING \
                --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --ndepths=48 --CostNum=4 --iters=12 --GRUiters="3,3,3" | tee -i $dirAndName

#finetune on BlendedMVS
#python -u train_bld.py --mode='train' --epochs=16 --numdepth=768 --trainviews=9 --logdir $LOG_DIR --loadckpt $LOG_DIR_CKPT --testviews=11 --dataset=blend --batch_size=2 --trainpath=$BLEND_TRAINING \
#                --trainlist lists/blendedmvs/train.txt --testlist lists/blendedmvs/val.txt --ndepths=96 --CostNum=4 --lr=0.0002 --iters=12 --GRUiters="3,3,3" | tee -i $dirAndName