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

python /home/selim/ME-GraphAU-main/test_link_prediction.py --dataset BP4D --arc resnet50 --exp-name test_fold3  --resume results/resnet50_second_stage/bs_64_seed_0_lr_1e-05/epoch48_model_fold4.pth --fold 3 --gnn_type GAT --feed_type vertex --root_dir /home/selim/ME-GraphAU-main/


echo "DONE!!!!"
