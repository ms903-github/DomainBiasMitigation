#!/bin/bash

#initialize modules
source /etc/profile.d/modules.sh

#modules
module load python/3.6
module load cuda/10.0
module load cudnn/7.4/7.4.2
module load nccl/2.3/2.3.5-2
module load openmpi/2.1.5
source work/bin/activate
# env SGE_O_WORKDIR = /home/acb11949pt/adience-fair-classify
cd "/home/acb11949pt/DomainBiasMitigation"

# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_baseline" --train_data "data/imdb_txt/imdb_tr_ex2_1_1.txt" --experiment_name "ex2_1_1" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_sampling" --train_data "data/imdb_txt/imdb_tr_ex2_1_1_sampled.txt" --experiment_name "ex2_1_1" &
# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_domain_discriminative0" --train_data "data/imdb_txt/imdb_tr_ex2_1_1.txt" --experiment_name "ex2_1_1" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_1_1.txt" --experiment_name "ex2_1_1" &
# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_uniconf_adv" --train_data "data/imdb_txt/imdb_tr_ex2_1_1.txt" --experiment_name "ex2_1_1" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_gradproj_adv" --train_data "data/imdb_txt/imdb_tr_ex2_1_1.txt" --experiment_name "ex2_1_1" &

# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_baseline" --train_data "data/imdb_txt/imdb_tr_ex2_1_2.txt" --experiment_name "ex2_1_2" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_sampling" --train_data "data/imdb_txt/imdb_tr_ex2_1_2_sampled.txt" --experiment_name "ex2_1_2" &
# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_domain_discriminative1" --train_data "data/imdb_txt/imdb_tr_ex2_1_2.txt" --experiment_name "ex2_1_2" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_1_2.txt" --experiment_name "ex2_1_2" &
# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_uniconf_adv" --train_data "data/imdb_txt/imdb_tr_ex2_1_2.txt" --experiment_name "ex2_1_2" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_gradproj_adv" --train_data "data/imdb_txt/imdb_tr_ex2_1_2.txt" --experiment_name "ex2_1_2" &

# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_baseline" --train_data "data/imdb_txt/imdb_tr_ex2_1_3.txt" --experiment_name "ex2_1_3" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_sampling" --train_data "data/imdb_txt/imdb_tr_ex2_1_3_sampled.txt" --experiment_name "ex2_1_3" &
# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_domain_discriminative2" --train_data "data/imdb_txt/imdb_tr_ex2_1_3.txt" --experiment_name "ex2_1_3" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_1_3.txt" --experiment_name "ex2_1_3" & 
# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_uniconf_adv" --train_data "data/imdb_txt/imdb_tr_ex2_1_3.txt" --experiment_name "ex2_1_3" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_gradproj_adv" --train_data "data/imdb_txt/imdb_tr_ex2_1_3.txt" --experiment_name "ex2_1_3" &

# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_baseline" --train_data "data/imdb_txt/imdb_tr_ex2_1_4.txt" --experiment_name "ex2_1_4" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_sampling" --train_data "data/imdb_txt/imdb_tr_ex2_1_4_sampled.txt" --experiment_name "ex2_1_4" &
# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_domain_discriminative3" --train_data "data/imdb_txt/imdb_tr_ex2_1_4.txt" --experiment_name "ex2_1_4" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_1_4.txt" --experiment_name "ex2_1_4" &
# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_uniconf_adv" --train_data "data/imdb_txt/imdb_tr_ex2_1_4.txt" --experiment_name "ex2_1_4" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_gradproj_adv" --train_data "data/imdb_txt/imdb_tr_ex2_1_4.txt" --experiment_name "ex2_1_4" &

# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_baseline" --train_data "data/imdb_txt/imdb_tr_ex2_2_1.txt" --experiment_name "ex2_2_1" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_sampling" --train_data "data/imdb_txt/imdb_tr_ex2_2_1_sampled.txt" --experiment_name "ex2_2_1" &
# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_domain_discriminative4" --train_data "data/imdb_txt/imdb_tr_ex2_2_1.txt" --experiment_name "ex2_2_1" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_2_1.txt" --experiment_name "ex2_2_1" &
# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_uniconf_adv" --train_data "data/imdb_txt/imdb_tr_ex2_2_1.txt" --experiment_name "ex2_2_1" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_gradproj_adv" --train_data "data/imdb_txt/imdb_tr_ex2_2_1.txt" --experiment_name "ex2_2_1" &

# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_baseline" --train_data "data/imdb_txt/imdb_tr_ex2_2_2.txt" --experiment_name "ex2_2_2" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_sampling" --train_data "data/imdb_txt/imdb_tr_ex2_2_2_sampled.txt" --experiment_name "ex2_2_2" &
# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_domain_discriminative4" --train_data "data/imdb_txt/imdb_tr_ex2_2_2.txt" --experiment_name "ex2_2_2" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_2_2.txt" --experiment_name "ex2_2_2" &
# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_uniconf_adv" --train_data "data/imdb_txt/imdb_tr_ex2_2_2.txt" --experiment_name "ex2_2_2" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_gradproj_adv" --train_data "data/imdb_txt/imdb_tr_ex2_2_2.txt" --experiment_name "ex2_2_2" &

# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_baseline" --train_data "data/imdb_txt/imdb_tr_ex2_2_3.txt" --experiment_name "ex2_2_3" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_sampling" --train_data "data/imdb_txt/imdb_tr_ex2_2_3_sampled.txt" --experiment_name "ex2_2_3" &
# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_domain_discriminative4" --train_data "data/imdb_txt/imdb_tr_ex2_2_3.txt" --experiment_name "ex2_2_3" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_2_3.txt" --experiment_name "ex2_2_3" &
# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_uniconf_adv" --train_data "data/imdb_txt/imdb_tr_ex2_2_3.txt" --experiment_name "ex2_2_3" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_gradproj_adv" --train_data "data/imdb_txt/imdb_tr_ex2_2_3.txt" --experiment_name "ex2_2_3" &

# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_baseline" --train_data "data/imdb_txt/imdb_tr_ex2_3.txt" --experiment_name "ex2_3" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_sampling" --train_data "data/imdb_txt/imdb_tr_ex2_3_sampled.txt" --experiment_name "ex2_3" &
# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_domain_discriminative4" --train_data "data/imdb_txt/imdb_tr_ex2_3.txt" --experiment_name "ex2_3" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_3.txt" --experiment_name "ex2_3" &
# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_uniconf_adv" --train_data "data/imdb_txt/imdb_tr_ex2_3.txt" --experiment_name "ex2_3" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_gradproj_adv" --train_data "data/imdb_txt/imdb_tr_ex2_3.txt" --experiment_name "ex2_3" &

# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_baseline" --train_data "data/imdb_txt/imdb_tr_ex2_4.txt" --experiment_name "ex2_4" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_sampling" --train_data "data/imdb_txt/imdb_tr_ex2_4_sampled.txt" --experiment_name "ex2_4" &
# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_domain_discriminative4" --train_data "data/imdb_txt/imdb_tr_ex2_4.txt" --experiment_name "ex2_4" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_4.txt" --experiment_name "ex2_4" &
# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_uniconf_adv" --train_data "data/imdb_txt/imdb_tr_ex2_4.txt" --experiment_name "ex2_4" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_gradproj_adv" --train_data "data/imdb_txt/imdb_tr_ex2_4.txt" --experiment_name "ex2_4" &

CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex1_1_sampled.txt" --experiment_name "sampled_ex1_1" &
CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex1_2_sampled.txt" --experiment_name "sampled_ex1_2" &
CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex1_3_sampled.txt" --experiment_name "sampled_ex1_3" &
CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex1_4_sampled.txt" --experiment_name "sampled_ex1_4" &
# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex1_5_sampled.txt" --experiment_name "sampled_ex1_5" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_1_1_sampled.txt" --experiment_name "sampled_ex2_1_1" &
# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_1_2_sampled.txt" --experiment_name "sampled_ex2_1_2" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_1_3_sampled.txt" --experiment_name "sampled_ex2_1_3" &
# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_1_4_sampled.txt" --experiment_name "sampled_ex2_1_4" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_2_1_sampled.txt" --experiment_name "sampled_ex2_2_1" &
# CUDA_VISIBLE_DEVICES=2 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_2_2_sampled.txt" --experiment_name "sampled_ex2_2_2" &
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_2_3_sampled.txt" --experiment_name "sampled_ex2_2_3" &
# CUDA_VISIBLE_DEVICES=0 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_3_sampled.txt" --experiment_name "sampled_ex2_3" &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment "imdb_domain_independent" --train_data "data/imdb_txt/imdb_tr_ex2_4_sampled.txt" --experiment_name "sampled_ex2_4" &

wait
echo "all process finished"