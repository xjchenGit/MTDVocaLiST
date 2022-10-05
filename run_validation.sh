setting_file="setting_SADW_II_xbm_SAD_cos_wrestart_lw_xbm_warmup_thin_ablation"
# note_name="layer_select_exp_stuv2_200_TransDistil_all_T1_alpha1_combine_f3_av4_va1_T25_transD_fitnet_star_bu"
# note_name="layer_select_exp_stuv2_200_TransDistil_all_T1_alpha1_combine_f3_av4_va1_T25_save_every_epoch"
# note_name="layer_select_exp_stuv2_200_TransDistil_all_T1_alpha1_combine_f3_av4_va1_T25_onlyrep_save_epoch"
note_name="exp_student_thin_200dim_FitNet_both_emb_star"
loss_fuc="transdistill_layer_select_clip_c3"
loss_subcfg="transdistill_layer_select_f3_clip_f3_av4_va1_T25"

python3 validation.py \
--note ${note_name} \
--experiment_dir /work/ntuvictor98/lip-sync/vocalist_v2/${setting_file}/ \
--teacher_checkpoint /work/ntuvictor98/lip-sync/vocalist/experiments/vocalist_5f_lrs2_repro_pt_offical/Best.pth \
--loss_config /work/ntuvictor98/lip-sync/vocalist_v2/${setting_file}/dzoo_transdistil.ini \
--loss_function ${loss_fuc} \
--loss_subcfg ${loss_subcfg} \
--epoch 80

# --student_checkpoint /work/ntuvictor98/lip-sync/vocalist_v2/setting_SADW_II_xbm_SAD_cos_wrestart_lw_xbm_warmup/_exp_sadwiism_0.001_T5_sad_0.05_0.5_lw_xbm_warmup_stuv2/Last.pth \
# --student_checkpoint /work/ntuvictor98/lip-sync/vocalist/experiments/vocalist_5f_lrs2_student7.5m/Best.pth \