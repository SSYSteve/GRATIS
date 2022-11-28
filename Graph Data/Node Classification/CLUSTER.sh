python main_SBMs_node_classification_best_model.py --dataset SBM_CLUSTER \
--gpu_id 1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_500k.json' \
--batch_size 16 --out_dir ./output/backbone/CLUSTER/gat_2x/ \
--dropout 0.1 --max_time 60 \
> ./output/backbone/CLUSTER/gat_2x/train.log 2>&1

#--dropout 0.1 --max_time 60 \
#--batch_size 128

# SBMs_node_clustering_GatedGCN_CLUSTER_PE_500k

# SBMs_node_clustering_GAT_CLUSTER_500k
# SBMs_node_clustering_GatedGCN_CLUSTER_PE_500k
# SBMs_node_clustering_GatedGCN_CLUSTER_500k

