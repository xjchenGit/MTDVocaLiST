# MTDVocaLiST


## 1.Train a teacher model

```
bash run_train.sh
```

## 2. Train a student model (without distillation)
```
bash run_train_student_thin.sh
```

## 3. Train a student model (with MTD loss)
```
bash run_train_student_thin_trans_distil_all_layer_selection_combine_f3_av4_va1_T25_save_every_epoch.sh
```

## 4. Evaluation
```
torch_test_lrs2_with_ext_bb_stu.py --checkpoint_path xxx/xxx.pth
```