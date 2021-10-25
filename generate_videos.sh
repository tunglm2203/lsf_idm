#!/usr/bin/env bash


# TODO: Cartpole
LOGDIR="videos/cartpole-swingup"
N_TESTS=5

#STEP=12500
STEP=62500


mkdir -p ${LOGDIR}

#python run_policy_dmcontrol.py --dir logs/cartpole-swingup/sac_lsf_rad_1000init_1gs_aclr1e-3_clr1e-3_allr1e-4_elr1e-3_ac_freq2_2_enccri_tau0.05_0.01_declr1e-3_bs128_baseline_extra_upd_critic_uselsf_1_extraupd_1_n_invupd_1_aug_0_alpha0.5/baseline_extra_upd_critic_uselsf_1_extraupd_1_n_invupd_1_aug_0_alpha0.5/pix84-s2-09_07-23_41_57/ \
#  --record --logdir ${LOGDIR} --n_tests ${N_TESTS} \
#  --step ${STEP}

#python run_policy_dmcontrol.py --dir logs/cartpole-swingup/sac_lsf_rad_1000init_1gs_aclr1e-4_clr1e-4_allr1e-4_elr1e-4_ac_freq2_2_enccri_tau0.05_0.01_declr1e-4_bs128_baseline_extra_upd_critic_uselsf_0_extraupd_1_n_invupd_1_aug_0_alpha0.5/baseline_extra_upd_critic_uselsf_0_extraupd_1_n_invupd_1_aug_0_alpha0.5/pix84-s2-09_01-07_50_06/ \
#  --record --logdir ${LOGDIR} --n_tests ${N_TESTS} \
#  --step ${STEP}

python run_policy_dmcontrol.py --dir logs/cartpole-swingup/lsf_previous_method/baseline_extra_upd_critic_uselsf_1_extraupd_1_n_invupd_1_aug_0/pix84-s1-08_31-21_19_59/ \
--record --logdir ${LOGDIR} --n_tests ${N_TESTS} \
--step ${STEP}


# TODO: Cheetah
#LOGDIR="videos/cheetah-run"
#N_TESTS=5
#
##STEP=25000
#STEP=125000
#
#
#mkdir -p ${LOGDIR}
#
##python run_policy_dmcontrol.py --dir logs/cheetah-run/sac_lsf_rad_1000init_1gs_aclr1e-3_clr1e-3_allr1e-4_elr1e-3_ac_freq2_2_enccri_tau0.05_0.01_declr1e-3_bs128_baseline_extra_upd_critic_uselsf_1_extraupd_1_n_invupd_1_aug_0/baseline_extra_upd_critic_uselsf_1_extraupd_1_n_invupd_1_aug_0/pix84-s2-09_03-19_03_28/ \
##  --record --logdir ${LOGDIR} --n_tests ${N_TESTS} \
##  --step ${STEP}
#
##python run_policy_dmcontrol.py --dir logs/cheetah-run/sac_lsf_rad_1000init_1gs_aclr1e-4_clr1e-4_allr1e-4_elr1e-4_ac_freq2_2_enccri_tau0.05_0.01_declr1e-4_bs128_baseline_extra_upd_critic_uselsf_0_extraupd_1_n_invupd_1_aug_0/baseline_extra_upd_critic_uselsf_0_extraupd_1_n_invupd_1_aug_0/pix84-s1-08_31-12_24_56/ \
##  --record --logdir ${LOGDIR} --n_tests ${N_TESTS} \
##  --step ${STEP}
#
#python run_policy_dmcontrol.py --dir logs/cheetah-run/lsf_previous_method/pix84-s2-08_28-21_48_17/ \
#  --record --logdir ${LOGDIR} --n_tests ${N_TESTS} \
#  --step ${STEP}





# TODO: Manipulation





