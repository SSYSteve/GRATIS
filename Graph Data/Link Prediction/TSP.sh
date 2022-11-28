#Gated-GCN base
python main_TSP_edge_classification_best_model.py --dataset TSP \
--gpu_id 0 \
--config 'configs/TSP_edge_classification_GatedGCN_100k.json' --edge_feat True \
--batch_size 16 \
--max_time 60 \

#GAT base
python main_TSP_edge_classification_best_model.py --dataset TSP \
--gpu_id 1 \
--config 'configs/TSP_edge_classification_GAT_edgereprfeat.json' --edge_feat True \
--batch_size 16 \
--max_time 60 \




