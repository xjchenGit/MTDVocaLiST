python3 train_vocalist_lrs2.py --note exp \
                                --experiment_dir /work/ntuvictor98/lip-sync/vocalist_v2/setting_SADW_inter_intra/ \
                                --teacher_checkpoint /work/ntuvictor98/lip-sync/vocalist/experiments/vocalist_5f_lrs2_repro_pt_offical/Best.pth \
                                --student_checkpoint /work/ntuvictor98/lip-sync/vocalist/experiments/vocalist_5f_lrs2_student7.5m/Best.pth \
                                --loss_config /work/ntuvictor98/lip-sync/vocalist_v2/setting_SADW_inter_intra/dzoo.ini \
                                --loss_function SADWII \
                                --alpha 0.0001
                                # --debug