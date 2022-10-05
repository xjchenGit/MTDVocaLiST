setting_file="setting_SADW_II_xbm_SAD_cos_wrestart_lw_xbm_warmup_thin"

python3 train_vocalist_lrs2_stuv2_xbm.py \
--note exp_sadwiism_T10_sad_33_lw_xbm_warmup_stuv2_ps_diff_xbm8176_thin_200_resume \
--experiment_dir /work/ntuvictor98/lip-sync/vocalist_v2/${setting_file}/ \
--teacher_checkpoint /work/ntuvictor98/lip-sync/vocalist/experiments/vocalist_5f_lrs2_repro_pt_offical/Best.pth \
--loss_config /work/ntuvictor98/lip-sync/vocalist_v2/${setting_file}/dzoo.ini \
--loss_function SADWIIsm_SAD_cross_SAD_kl \
--alpha 3 \
--beta 3

# --student_checkpoint /work/ntuvictor98/lip-sync/vocalist_v2/setting_SADW_II_xbm_SAD_cos_wrestart_lw_xbm_warmup_thin/_exp_sadwiism_T10_sad_33_lw_xbm_warmup_stuv2_ps_diff_xbm8176_thin_200_resume/Best_94.07.pth \
# --xbm_size 4086
# --debug
# --student_checkpoint /work/ntuvictor98/lip-sync/vocalist_v2/setting_SADW_II_xbm_SAD_cos_wrestart_lw_xbm_warmup/_exp_sadwiism_0.001_T5_sad_0.05_0.5_lw_xbm_warmup_stuv2/Last.pth \
# --student_checkpoint /work/ntuvictor98/lip-sync/vocalist/experiments/vocalist_5f_lrs2_student7.5m/Best.pth \
