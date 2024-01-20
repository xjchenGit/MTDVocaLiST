# MTDVocaLiST
We proposed an lightweight audio-visual synchronization (AVS)model **[MTDVocaLiST](https://arxiv.org/abs/2210.15563)**. MTDVocaLiST reduces the model size of **[VocaLiST](https://github.com/vskadandale/vocalist)** by 83.52%, yet still maintaining similar performance. Audio-visual synchronization aims to determine whether the mouth movements and speech in the video are synchronized. This repository is the official repository for the paper
[Multimodal Transformer Distillation for Audio-Visual Synchronization](https://arxiv.org/abs/2210.15563). The paper had been accepted by ICASSP 2024.

## Datasets and preprocessing
There are 2 datasets involved in this work: i) The AV speech dataset of LRS2, and ii) the AV singing voice dataset of Acappella. The LRS2 dataset can be requested for download [here](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html). All the models considered in this work operate on audios sampled at 16kHz and videos with 25fps. The preprocessing steps are the same as [Wav2Lip](https://github.com/Rudrabha/Wav2Lip/blob/master/preprocess.py). The objective of the preprocessing step is to obtain the cropped RGB face frames of size 3x96x96 in the .jpg format and audios of 16kHz sampling rate for each of the video samples in respective datasets.

## Leverage pre-trained MTDVocaLiST only

```python
import torch
from models.student_thin_200_all import SyncTransformer

cpk = torch.load("pretrained/pure_MTDVocaLiST.pth", map_location='cpu')
model = SyncTransformer(d_model=200)
model.load_state_dict(cpk["state_dict"])
```

## Training (Multimodal Transformer Distillation)
You need to download the pretrained VocaLiST model firstly from [[Weights]](https://drive.google.com/drive/folders/1-g4qHUNNcCZpmSqEflKMxPMvwnn9e88N?usp=sharing).

```bash
bash run_train_student.sh
```

## Evaluation (Inference)

```
python3 test_stu.py --data_root /path/to/lip-sync/LRS2_wav2lip/main/ --checkpoint_path /path/to/Best.pth
```

## Comparison with SOTA AVS model

<div class="center" style="text-align: center">
    <div class="center col-md-8" style="text-align: center">
        <img src="figures/size_and_acc.jpg"/>
    </div>
</div>

## Citation
If you find our work useful, please consider cite
```
@article{chen2022multimodal,
  title={Multimodal Transformer Distillation for Audio-Visual Synchronization},
  author={Chen, Xuanjun and Wu, Haibin and Wang, Chung-Che and Lee, Hung-yi and Jang, Jyh-Shing Roger},
  journal={arXiv preprint arXiv:2210.15563},
  year={2022}
}

@article{kadandale2022vocalist,
  title={Vocalist: An audio-visual synchronisation model for lips and voices},
  author={Kadandale, Venkatesh S and Montesinos, Juan F and Haro, Gloria},
  journal={arXiv preprint arXiv:2204.02090},
  year={2022}
}
```
## Acknowledgement
If you have any question, please feel free to contact with me by email d12942018@ntu.edu.tw.