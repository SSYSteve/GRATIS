#!/bin/bash


cd Non-Graph
cd Node-classification-IJCAI


# to train the first stage of our approach (ResNet-50) on BP4D Dataset, run:
python train_stage1.py --dataset BP4D --arc resnet50 --exp-name resnet50_first_stage -b 64 -lr 0.0001 --fold 1 
#to train the second stage of our approach (ResNet-50) on BP4D Dataset, run:
python train_stage2.py --dataset BP4D --arc resnet50 --exp-name resnet50_second_stage  --resume results/resnet50_first_stage/bs_64_seed_0_lr_0.0001/xxxx_fold1.pth --fold 1 --lam 0.05
#to train the first stage of our approach (Swin-B) on DISFA Dataset, run:
python train_stage1.py --dataset DISFA --arc swin_transformer_base --exp-name swin_transformer_base_first_stage -b 64 -lr 0.0001 --fold 2
# to train the second stage of our approach (Swin-B) on DISFA Dataset, run:
python train_stage2.py --dataset DISFA --arc swin_transformer_base --exp-name swin_transformer_base_second_stage  --resume results/swin_transformer_base_first_stage/bs_64_seed_0_lr_0.0001/xxxx_fold2.pth -b 64 -lr 0.000001 --fold 2 --lam 0.01 
# to test the performance on DISFA Dataset, run:
python test.py --dataset DISFA --arc swin_transformer_base --exp-name test_fold2  --resume results/swin_transformer_base_second_stage/bs_64_seed_0_lr_0.000001/xxxx_fold2.pth --fold 2
