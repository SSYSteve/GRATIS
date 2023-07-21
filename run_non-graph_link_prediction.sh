#!/bin/bash


cd Non-Graph
cd Link-Prediction-AU
# to train our approach (ResNet-50) on BP4D Dataset with Gated-GCN and only vertex feeding to final classifier, run:
python train_link_prediction.py --dataset BP4D --arc resnet50 --exp-name resnet50_link_prediction  --resume results/resnet50_link_prediction/bs_64_seed_0_lr_0.0001/xxxx_fold1.pth --fold 1 --lam 0.05 --gnn_type GCN --feed_type vertex  --root_dir .
# to train our approach (Swin-B) on BP4D Dataset with GAT and both edge and vertex feeding to final classifier, run:
python train_link_prediction.py --dataset BP4D --arc swin_transformer_base --exp-name swin_base_link_prediction --resume results/swin_base_link_prediction/bs_64_seed_0_lr_0.0001/xxxx_fold1.pth --fold 1 --lam 0.05 --gnn_type GAT --feed_type vertex+edge  --root_dir .

# to train our approach (ResNet-50) on DISFA Dataset with Gated-GCN and edge feeding to final classifier, run:
python train_link_prediction.py --dataset DISFA --arc swin_transformer_base --exp-name swin_base_link_prediction  --resume results/swin_base_link_prediction/bs_64_seed_0_lr_0.0001/xxxx_fold1.pth --fold 1 --lam 0.05 --gnn_type GAT --feed_type edge    --root_dir .
# to test the performance on DISFA Dataset (for Swin-B, Gated-GCN and edge feed), run:
python test_link_prediction.py --dataset DISFA --arc swin_transformer_base --exp-name test_fold2  --resume results/swin_transformer_base_link_prediction/bs_64_seed_0_lr_0.000001/xxxx_fold2.pth --fold 2 --gnn_type GCN --feed_type edge  --root_dir .