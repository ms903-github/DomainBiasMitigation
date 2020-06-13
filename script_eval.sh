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

echo baseline
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex1_1/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex1_1/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex1_2/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex1_2/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex1_3/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex1_3/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex1_4/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex1_4/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex1_5/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex1_5/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex2_1_1/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex2_1_1/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex2_1_2/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex2_1_2/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex2_1_3/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex2_1_3/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex2_1_4/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex2_1_4/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex2_2_1/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex2_2_1/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex2_2_2/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex2_2_2/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex2_2_3/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex2_2_3/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex2_3/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex2_3/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_baseline/ex2_4/test_male_result.pkl" --f_load_path "record/imdb_baseline/ex2_4/test_female_result.pkl"
echo sampling
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex1_1/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex1_1/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex1_2/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex1_2/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex1_3/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex1_3/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex1_4/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex1_4/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex1_5/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex1_5/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex2_1_1/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex2_1_1/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex2_1_2/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex2_1_2/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex2_1_3/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex2_1_3/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex2_1_4/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex2_1_4/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex2_2_1/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex2_2_1/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex2_2_2/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex2_2_2/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex2_2_3/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex2_2_3/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex2_3/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex2_3/test_female_result.pkl"
python eval.py --mode "baseline" --m_load_path "record/imdb_sampling/ex2_4/test_male_result.pkl" --f_load_path "record/imdb_sampling/ex2_4/test_female_result.pkl"
# echo domain_discriminative
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative0/ex1_1/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative0/ex1_1/test_female_result.pkl"
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative1/ex1_2/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative1/ex1_2/test_female_result.pkl"
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative2/ex1_3/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative2/ex1_3/test_female_result.pkl"
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative3/ex1_4/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative3/ex1_4/test_female_result.pkl"
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative4/ex1_5/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative4/ex1_5/test_female_result.pkl"
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative/ex2_1_1/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative/ex2_1_1/test_female_result.pkl"
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative/ex2_1_2/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative/ex2_1_2/test_female_result.pkl"
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative/ex2_1_3/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative/ex2_1_3/test_female_result.pkl"
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative/ex2_1_4/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative/ex2_1_4/test_female_result.pkl"
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative/ex2_2_1/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative/ex2_2_1/test_female_result.pkl"
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative/ex2_2_2/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative/ex2_2_2/test_female_result.pkl"
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative/ex2_2_3/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative/ex2_2_3/test_female_result.pkl"
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative/ex2_3/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative/ex2_3/test_female_result.pkl"
# python eval.py --mode "domain_discriminative" --m_load_path "record/imdb_domain_discriminative/ex2_4/test_male_result.pkl" --f_load_path "record/imdb_domain_discriminative/ex2_4/test_female_result.pkl"
echo domain_independent
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex1_1/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex1_1/test_female_result.pkl"
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex1_2/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex1_2/test_female_result.pkl"
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex1_3/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex1_3/test_female_result.pkl"
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex1_4/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex1_4/test_female_result.pkl"
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex1_5/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex1_5/test_female_result.pkl"
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex2_1_1/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex2_1_1/test_female_result.pkl"
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex2_1_2/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex2_1_2/test_female_result.pkl"
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex2_1_3/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex2_1_3/test_female_result.pkl"
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex2_1_4/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex2_1_4/test_female_result.pkl"
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex2_2_1/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex2_2_1/test_female_result.pkl"
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex2_2_2/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex2_2_2/test_female_result.pkl"
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex2_2_3/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex2_2_3/test_female_result.pkl"
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex2_3/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex2_3/test_female_result.pkl"
python eval.py --mode "domain_independent" --m_load_path "record/imdb_domain_independent/ex2_4/test_male_result.pkl" --f_load_path "record/imdb_domain_independent/ex2_4/test_female_result.pkl"
echo gradproj_adv
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex1_1/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex1_1/test_female_result.pkl"
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex1_2/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex1_2/test_female_result.pkl"
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex1_3/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex1_3/test_female_result.pkl"
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex1_4/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex1_4/test_female_result.pkl"
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex1_5/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex1_5/test_female_result.pkl"
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex2_1_1/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex2_1_1/test_female_result.pkl"
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex2_1_2/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex2_1_2/test_female_result.pkl"
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex2_1_3/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex2_1_3/test_female_result.pkl"
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex2_1_4/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex2_1_4/test_female_result.pkl"
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex2_2_1/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex2_2_1/test_female_result.pkl"
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex2_2_2/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex2_2_2/test_female_result.pkl"
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex2_2_3/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex2_2_3/test_female_result.pkl"
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex2_3/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex2_3/test_female_result.pkl"
python eval.py --mode "gradproj_adv" --m_load_path "record/imdb_gradproj_adv/ex2_4/test_male_result.pkl" --f_load_path "record/imdb_gradproj_adv/ex2_4/test_female_result.pkl"

echo uniconf_adv
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex1_1/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex1_1/test_female_result.pkl"
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex1_2/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex1_2/test_female_result.pkl"
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex1_3/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex1_3/test_female_result.pkl"
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex1_4/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex1_4/test_female_result.pkl"
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex1_5/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex1_5/test_female_result.pkl"
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex2_1_1/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex2_1_1/test_female_result.pkl"
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex2_1_2/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex2_1_2/test_female_result.pkl"
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex2_1_3/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex2_1_3/test_female_result.pkl"
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex2_1_4/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex2_1_4/test_female_result.pkl"
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex2_2_1/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex2_2_1/test_female_result.pkl"
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex2_2_2/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex2_2_2/test_female_result.pkl"
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex2_2_3/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex2_2_3/test_female_result.pkl"
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex2_3/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex2_3/test_female_result.pkl"
python eval.py --mode "uniconf_adv" --m_load_path "record/imdb_uniconf_adv/ex2_4/test_male_result.pkl" --f_load_path "record/imdb_uniconf_adv/ex2_4/test_female_result.pkl"
