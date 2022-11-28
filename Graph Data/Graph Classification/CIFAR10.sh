#Gated-GCN base
python main_superpixels_graph_classification_best_model.py --dataset CIFAR10 \
--gpu_id 1 \
--config 'configs/superpixels_graph_classification_GatedGCN_CIFAR10_100k.json' --batch_size 32 \
--dropout 0.1 --max_time 120 \

#GAT base
python main_superpixels_graph_classification_best_model.py --dataset CIFAR10 \
--gpu_id 0 \
--config 'configs/superpixels_graph_classification_GAT_CIFAR10_100k.json' --batch_size 64 \
--dropout 0.1 --max_time 120 \
