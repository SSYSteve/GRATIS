#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=expression                    # job name
#SBATCH --workdir=/home/selim/ # working directory
#SBATCH --gres=gpu:1                            # how many gpu you want.
#SBATCH --output=/home/selim/outs_errors/slurm-%j.out   # output file
#SBATCH --error=/home/selim/outs_errors/slurm-%j.err    # output error file
#SBATCH --time=15-00:00:00                              # maximum running time. 
#SBATCH --nodelist=romer2

########### CUDA Paths. You can change or delete them. #############
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
#########################################################

python3 /home/selim/ME-GraphAU-main/train_link_prediction.py --dataset BP4D --arc resnet50 --exp-name resnet_to_report_nodes_g-gcn_bp4d_link_pred --fold 3 --gnn_type GCN --feed_type vertex --epochs 50 --root_dir /home/selim/ME-GraphAU-main/ --batch-size 64 --learning-rate 0.000001 --weight-decay 0.000005

echo "DONE!!!!"
