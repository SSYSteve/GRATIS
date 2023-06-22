# GRATIS: Deep Learning Graph Representation with Task-specific Topology and Multi-dimensional Edge Features

This is the official code for the paper: GRATIS: Deep Learning Graph Representation with Task-specific Topology and Multi-dimensional Edge Features
## Graph Data

<!---![pipeline_graph](/figures/pipeline_graph.png)--->
<img src="/figures/pipeline_graph.png" alt="drawing" width="700"/>

Six graph datasets are employed (two for each task): (i) Graph classification: the MNIST and CIFAR10 datasets are employed; (ii) Vertex (node) classification: the PATTERN and CLUSTER datasets are employed; and (iii) Edge link prediction: the TSP and COLLAB datasets are employed.

### Graph Classification (Graph)


#### Data Preparation

Download the MNIST and CIFAR10 Super-pixel datasets and preprocess the data with respectively the given notebooks on this webpage https://github.com/graphdeeplearning/benchmarking-gnns/tree/master/data/superpixels: i.e., ```prepare_superpixels_CIFAR.ipynb and prepare_superpixels_MNIST.ipynb``` or directly use https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/script_download_superpixels.sh instead.

The given notebook includes the training, validation, and testing data splitting and will obtain a new ```pkl``` file. In other words, two files (data/superpixels/CIFAR10.pkl and data/superpixels/MNIST.pkl) will be generated after this step.

#### Training Command

We have included the training command in respectively the ```MNIST.sh and CIFAR10.sh``` two bash files, where we present the detailed training command for each method, for example:
`python main_superpixels_graph_classification_best_model.py --dataset MNIST --gpu_id 0 --config 'configs/superpixels_graph_classification_GatedGCN_MNIST_100k.json' --batch_size 32 --dropout 0.1 --max_time 120 \`


#### Data Loading

In the **main_superpixels_graph_classification_best_model.py**, we set up the training environments, such as the optimizer, learning rate scheduler, and data loaders.

```python

optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=params['lr_reduce_factor'],patience=params['lr_schedule_patience'],verbose=True)

train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

```
#### Testing
The detailed training/validation/testing code is included in the **train_superpixels_graph_classification.py**, such as:
```python
from train.train_superpixels_graph_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch,args)
epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch, args)
_, epoch_test_acc = evaluate_network(model, device, test_loader, epoch, args)
```

### Node Classification  (Graph)
#### Data Preparation

Download the ```SBM_PATTERN and SBM_CLUSTER``` datasets using https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/script_download_SBMs.sh

The bash file will automatically download the ```SBM_CLUSTER.pkl and SBM_PATTERN.pkl```

#### Training Command

We have included the training command in respectively the ```CLUSTER.sh and PATTERN.sh``` two bash files, where we presents the detailed training command for each method, for example:
```
python main_SBMs_node_classification_best_model.py --dataset SBM_CLUSTER --gpu_id 1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_500k.json' --batch_size 16 --out_dir ./output/backbone/CLUSTER/gat_2x/ --dropout 0.1 --max_time 60
```

```
python main_SBMs_node_classification_best_model.py --dataset SBM_PATTERN --gpu_id 0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_PE_500k.json' --batch_size 16 --max_time 60
```

### Link Prediction (Graph)
#### Data Preparation

Download the ```TSP``` datasets using https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/script_download_TSP.sh

#### Training Command

```
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
```


## Non-Graph Data

<!---![pipeline_nongraph](/figures/pipeline_nongraph.png)--->
<img src="/figures/pipeline_nongraph.png" alt="drawing" width="700"/>

In the manuscript, we tried and reported the proposed GRATIS on three different tasks with five different face image datasets. Specifically, for i) Graph Classification, the FER 2013 and RAF-DB facial expression recognition (FER) datasets are employed, where we produce a graph for each face image, and predict image-level facial expressions (i.e., seven class classification problem) based on the produced graph; for ii) Node Classification, the BP4D, and DISFA Facial Action Units (AUs) Recognition datasets are employed, where the task is to jointly predict multiple AUs’ activations from each face image; for iii) Link Prediction, the BP4D and DISFA datasets are again employed, where we aim to recognize the co-occurrence pattern between a pair of AUs (vertices), i.e., the edge pattern of the corresponding AUs.


## Updating 

[2022 Nov 28] Note the current code is implemented by different co-authors. In the following weeks, it will be changed accordingly to form a more consistent project.

[2022 Nov 29] Uploaded the code of two non-graph experiments.

[2023 Jun 22] Uploaded a brief tutorial.
