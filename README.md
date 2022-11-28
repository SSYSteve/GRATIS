# GRATIS: Deep Learning Graph Representation with Task-specific Topology and Multi-dimensional Edge Features

This is the official code for the paper: GRATIS: Deep Learning Graph Representation with Task-specific Topology and Multi-dimensional Edge Features

## Updating [2022 Nov 28]

Note the current code is implemented by different co-authors. In the following weeks, it will be changed accordingly to form a more consistent project.

## Non-Graph Data

<!---![pipeline_nongraph](/figures/pipeline_nongraph.png)--->
<img src="/figures/pipeline_nongraph.png" alt="drawing" width="700"/>

In the manuscirpt, we tried and reported the proposed GRATIS on three different tasks with five different face image datasets. Specifically, for i) Graph Classificaiton, the FER 2013 and RAF-DB facial expression recognition (FER) datasets are employed, where we produce a graph for each face image, and predict image-level facial expressions (i.e., seven class classification problem) based on the produced graph; for ii) Node Classification, the BP4D and DISFA Facial Action Units (AUs) Recognition datasets are employed, where the task is to jointly predict multiple AUs’ activation from each face image; for iii) Link Prediction, the BP4D and DISFA datasets are again employed, where we aim to recognize the co-occurrence pattern between a pair of AUs (vertices), i.e., the edge pattern of the corresponding AUs.

## Graph Data

<!---![pipeline_graph](/figures/pipeline_graph.png)--->
<img src="/figures/pipeline_graph.png" alt="drawing" width="700"/>

Six graph datasets are employed (two for each task): (i) Graph classification: the MNIST and CIFAR10 datasets are employed; (ii) Vertex (node) classification: the PATTERN and CLUSTER datasets are employed; and (iii) Edge link prediction: the TSP and COLLAB datasets are employed.

