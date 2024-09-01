# Preliminaries

In this section, we briefly introduce the basic concepts of graph representation as well as the general vertex and edge updating mechanism of Graph Neural Networks (GNNs).

## Graph Representation

A graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is composed of a set of vertices $\mathcal{V} \subseteq \{\mathbf{v}_i \in \mathbb{R}^{1 \times K}\}$, and edges $\mathcal{E} \subseteq \{\mathbf{e}_{i,j} = \mathbf{e}(\mathbf{v}_i, \mathbf{v}_j) \mid \mathbf{v}_i, \mathbf{v}_j \in \mathcal{V},  i \neq j\}$. Here, $\mathbf{v}_i$ represents $K$ attributes of the $i_{th}$ object/component in the predefined graph or non-graph data sample, and $\mathbf{e}_{i,j}$ represents the edge feature that defines the relationship between vertices $\mathbf{v}_i$ and $\mathbf{v}_j$. Each pair of vertices can be connected by at most one undirected edge or two directed edges.

A standard way to describe such edges is through the adjacency matrix $\mathcal{A} \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$, where all vertices in a graph are ordered so that each vertex indexes a specific row and column. The presence of each edge $\mathbf{e}_{i,j}$ is described by a binary value $\mathcal{A}_{i,j} = 1$ if $\mathbf{v}_i$ and $\mathbf{v}_j$ are connected, or $\mathcal{A}_{i,j} = 0$ otherwise. The adjacency matrix is symmetric if all edges are undirected but can be non-symmetric if one or more directed edges exist. Instead of using a binary value, some studies also build adjacency matrices with continuous real values to describe the strength of association between each pair of vertices.

## Vertex/Edge Updating of Message-Passing GNNs

Recently, message-passing Graph Neural Networks (GNNs), including Graph Convolution Networks (GCNs), have become dominant models for a wide variety of graph analysis tasks. Given a GNN $G$, its $l_{th}$ layer $G^l$ takes the graph $\mathcal{G}^{l-1} = (\mathcal{V}^{l-1}, \mathcal{E}^{l-1})$ produced by the $(l-1)_{th}$ layer $G^{l-1}$ as input and generates a new graph $\mathcal{G}^{l} = (\mathcal{V}^l, \mathcal{E}^l)$, which can be formulated as:

$
\mathcal{G}^{l} = G^l(\mathcal{G}^{l-1})
$

Specifically, the vertex feature $\mathbf{v}^l_i$ in $\mathcal{G}^{l}$ is computed based on: 
1. Its previous status $\mathbf{v}_i^{l-1}$ in $\mathcal{G}^{l-1}$.
2. A set of adjacent vertices $\mathbf{v}_j^{l-1} \subseteq \mathcal{N}(\mathbf{v}_i^{l-1})$ in $\mathcal{G}^{l-1}$, where $\mathcal{A}^{l-1}_{i,j} = 1$, and $\mathcal{A}^{l-1}$ is the adjacency matrix of $\mathcal{G}^{l-1}$.
3. A set of edge features $\mathbf{e}_{j,i}^{l-1}$ that represent the relationship between every $\mathbf{v}_j^{l-1}$ and $\mathbf{v}_i^{l-1}$ in $\mathcal{N}(\mathbf{v}_i^{l-1})$.

Here, the message $\mathbf{m}_{\mathcal{N}(\mathbf{v}^{l-1}_i)}$ is produced by aggregating all adjacent vertices of $\mathbf{v}_i^{l-1}$ through related edges $\mathbf{e}_{j,i}^{l-1}$, which can be formulated as:

\[
\mathbf{m}_{\mathcal{N}(\mathbf{v}^{l-1}_i)} = M\left(\mathbin\Vert ^{N}_{j=1} f(\mathbf{v}_j^{l-1}, \mathbf{e}_{j,i}^{l-1})\right)
\]
\[
f(\mathbf{v}_j^{l-1}, \mathbf{e}_{j,i}^{l-1}) = 0 \quad \text{subject to} \quad \mathcal{A}^{l-1}_{i,j} = 0
\]

where $M$ is a differentiable function that aggregates messages produced from all adjacent vertices; $N$ denotes the number of vertices in the graph $\mathcal{G}^{l-1}$; $f(\mathbf{v}_j^{l-1}, \mathbf{e}_{j,i}^{l-1})$ is a differentiable function defining the influence of an adjacent vertex $\mathbf{v}_j^{l-1}$ on the vertex $\mathbf{v}_i^{l-1}$ through their edge $\mathbf{e}_{j,i}^{l-1}$; and $\mathbin\Vert$ is the aggregation operator to combine messages of all adjacent vertices of $\mathbf{v}_i^{l-1}$. As a result, the vertex feature $\mathbf{v}^l_i$ can be updated as:

\[
\mathbf{v}^l_i = G_v^l(\mathbf{v}^{l-1}_i, \mathbf{m}_{\mathcal{N}(\mathbf{v}^{l-1}_i)})
\]

where $G_v^l$ denotes a differentiable function of the $l_{th}$ GNN layer, which updates all vertex features for producing the graph $\mathcal{G}^{l}$.

Meanwhile, each edge feature $\mathbf{e}_{i,j}^l$ in the graph $\mathcal{G}^{l}$ can be either kept the same as its previous status $\mathbf{e}_{i,j}^{l-1}$ in the graph $\mathcal{G}^{l-1}$ (denoted as the GNN type 1) or updated (denoted as the GNN type 2) during GNNs' propagation. Specifically, each edge feature $\mathbf{e}_{i,j}^l \in \mathcal{G}^{l}$ is computed based on:

1. Its previous status $\mathbf{e}^{l-1}_{i,j} \in \mathcal{G}^{l-1}$.
2. The corresponding vertex features $\mathbf{v}^{l-1}_i$ and $\mathbf{v}^{l-1}_j$ in $\mathcal{G}^{l-1}$.

Mathematically, $\mathbf{e}^{l}_{i,j}$ can be computed as:

\[
\mathbf{e}_{i,j}^l = 
\begin{cases}
\mathbf{e}^{l-1}_{i,j} &  \text{GNN type 1} \\
G_e^l(\mathbf{e}^{l-1}_{i,j}, g(\mathbf{v}^{l-1}_i, \mathbf{v}^{l-1}_j)) & \text{GNN type 2}
\end{cases}
\]

where $G_e^l$ is a differentiable function of the $l_{th}$ GNN layer, which updates edge features to produce the graph $\mathcal{G}^{l}$, and $g$ is a differentiable function that models the relationship between $\mathbf{v}^{l-1}_i$ and $\mathbf{v}^{l-1}_j$. In summary, vertex and edge features' updating are mutually influenced during the propagation of message-passing GNNs. Please refer to Hamilton et al. and Dwivedi et al. for more details.
