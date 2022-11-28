python main_COLLAB_edge_classification_best_model.py --dataset OGBL-COLLAB \
--gpu_id 0 --config 'configs/COLLAB_edge_classification_GatedGCN_40k.json' \
--dropout 0.1 --max_time 60 \

python main_COLLAB_edge_classification_best_model.py --dataset OGBL-COLLAB \
--gpu_id 1 --config 'configs/COLLAB_edge_classification_GAT_edgereprfeat.json' \
--out_dir ./output/test/ \
--dropout 0.1 --max_time 60 \
