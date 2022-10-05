setting_file="MTDVocaLiST"
note_name="layer_select_exp_stuv2_200_TransDistil_all_T1_alpha1_combine_f3_av4_va1_T25_save_every_epoch"
loss_fuc="transdistill_layer_select_clip_c3"
loss_subcfg="transdistill_layer_select_f3_clip_f3_av4_va1_T25"

python3 train_vocalist_lrs2_stuv2_TransDis_all_save_every_epoch.py \
--note ${note_name} \
--experiment_dir /work/ntuvictor98/lip-sync/vocalist_v2/${setting_file}/ \
--teacher_checkpoint /work/ntuvictor98/lip-sync/vocalist/experiments/vocalist_5f_lrs2_repro_pt_offical/Best.pth \
--loss_config /work/ntuvictor98/lip-sync/vocalist_v2/${setting_file}/dzoo_transdistil.ini \
--loss_function ${loss_fuc} \
--loss_subcfg ${loss_subcfg} \
--epoch 80
