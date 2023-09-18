##!/usr/bin/env bash

DTU_TESTING="/media/data1/datasets/dtu/dtu_test_mvsnet/"

TANK_TESTING='/media/data1/datasets/tanksandtemples_1/'

CKPT_FILE="checkpoints/blend_ft.ckpt"

OUT_DIR='outputs/dtu/'

#rt_cor_entropy_conf4 conf=0.4

if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
##DTU
python test.py --dataset=general_eval --batch_size=1 --testpath=$DTU_TESTING --ndepths=48 --CostNum=4 --numdepth=384 --GRUiters="3,3,3" --conf=0.5 --cost_reg="small" --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE --outdir $OUT_DIR --data_type dtu \
              --num_view=5
##tank  dtu_train
#python test.py --dataset=tank --batch_size=1 --testpath=$TANK_TESTING  --ndepths=96 --CostNum=4 --numdepth=384 --loadckpt $CKPT_FILE --outdir $OUT_DIR --data_type tank \
#              --num_view=11
##tank  blend_ft
#python test.py --dataset=tank --batch_size=1 --testpath=$TANK_TESTING  --ndepths=96 --CostNum=4 --numdepth=768 --loadckpt $CKPT_FILE --outdir $OUT_DIR --data_type tank \
#              --num_view=11
