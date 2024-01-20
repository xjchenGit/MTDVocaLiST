setting_file="experiments"
note_name="exp_200_TransDistil_all_layers_save_every_epoch"
loss_fuc="transdistill_full_layers_aw_loss_sqkv"
loss_subcfg="transdistill_all_layers_autoweight"

python3 train_MTD.py \
--note ${note_name} \
--experiment_dir ./${setting_file}/ \
--teacher_checkpoint pretrained/repro_Vocalist.pth \
--loss_config distillzoo.ini \
--loss_function ${loss_fuc} \
--loss_subcfg ${loss_subcfg} \
--epoch 80 \
