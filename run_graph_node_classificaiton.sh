#!/bin/bash


cd Graph\ Data/
cd Node\ Classification/


#GAT base
python main_SBMs_node_classification_best_model.py --dataset SBM_CLUSTER \
--gpu_id 1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_500k.json' \
--batch_size 16 --out_dir ./output/backbone/CLUSTER/gat_2x/ \
--dropout 0.1 --max_time 60 \
> ./output/backbone/CLUSTER/gat_2x/train.log 2>&1


#Gated-GCN base
python main_SBMs_node_classification_best_model.py --dataset SBM_PATTERN \
--gpu_id 0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_PE_500k.json' --batch_size 16 \
--max_time 60 \

#GAT base
python main_SBMs_node_classification_best_model.py --dataset SBM_PATTERN \
--gpu_id 1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_500k.json' --batch_size 32 \
--max_time 60 \