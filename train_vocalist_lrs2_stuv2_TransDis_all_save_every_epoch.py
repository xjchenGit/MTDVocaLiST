####################################################
# Adapted from https://github.com/Rudrabha/Wav2Lip #
####################################################
from os.path import dirname, join, basename, isfile
import torchaudio, torchvision
from tqdm import tqdm
import importlib

import warnings
warnings.filterwarnings("ignore")

from models.teacher_all import SyncTransformer as Tea_SyncTransformer
from models.student_thin_200_all import SyncTransformer as Stu_SyncTransformer

from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim
from torch.utils import data as data_utils
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as cvtransforms
from glob import glob
import os, random, argparse
from hparams import hparams, get_image_list
from natsort import natsorted
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from warmup_scheduler import GradualWarmupScheduler
### logging
from configparser import ConfigParser
from utils import get_logger, load_checkpoint
from utils import audio_feature_extractor

from distiller_zoo.FitNet import Loss as FitNetLoss

def parse_option():
    parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator on LRS2')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset",
                        default="/work/ntuvictor98/lip-sync/LRS2_wav2lip/pretrain/")
    parser.add_argument("--test_data_root", help="Root folder of the preprocessed LRS2 dataset",
                        default="/work/ntuvictor98/lip-sync/LRS2_wav2lip/main/")
    parser.add_argument('--teacher_checkpoint', help='Resumed from this checkpoint', default=None, type=str)
    parser.add_argument('--student_checkpoint', help='Resumed from this checkpoint', default=None, type=str)
    parser.add_argument('--experiment_dir', help='Save checkpoints to this directory', default=None, type=str)
    
    parser.add_argument('--loss_function', type=str, default='logit')
    parser.add_argument('--loss_config', type=str, default=None)
    parser.add_argument('--loss_subcfg', type=str, default=None)

    parser.add_argument('--learning_rate', help='learning rate', default=5e-5 ,type=float)
    parser.add_argument('--xbm_size', type=int, default=8172)
    parser.add_argument('--xbm_START_ITERATION', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_thread', type=int, default=24)
    parser.add_argument('--epoch', type=float, default=1000)
    parser.add_argument('--note', type=str, default=None)
    parser.add_argument('--debug', default='False', action='store_true', help='run to debug!')

    args = parser.parse_args()
    return args


def save_checkpoint(logger, hparams, model, optimizer, scheduler, step, checkpoint_dir, epoch, nepochs, better):

    if better:
        checkpoint_path = os.path.join(checkpoint_dir, "Best.pth")
    else:
        # checkpoint_path = os.path.join(checkpoint_dir, "Last.pth")
        checkpoint_path = os.path.join(checkpoint_dir, f"{ epoch + 1 }.pth")

    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    lr = optimizer.param_groups[0]['lr']
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        'scheduler': scheduler.state_dict(),
        "lr": lr
    }, checkpoint_path)
    if better:
        logger.info(f"Epoch: [{global_epoch + 1}/{nepochs}]\t Save best checkpoint, Better F1: {best_average_f1:.5f}")
    else:
        logger.info(f"Epoch: [{global_epoch + 1}/{nepochs}]\t Save last checkpoint!")

class Dataset(object):
    def __init__(self, split):
        self.split = split
        if split == 'pretrain':
            self.all_videos = get_image_list(args.data_root, split)
        else:
            self.all_videos = get_image_list(args.test_data_root, split)
        self.cvresize = cvtransforms.Resize([hparams.img_size, hparams.img_size])

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_wav(self, wavpath, vid_frame_id): # how to get vid_frame_id ?
        pos_aud_chunk_start = vid_frame_id * 640
        wav_vec, sr = torchaudio.load(wavpath, frame_offset=pos_aud_chunk_start, num_frames=num_audio_elements)
        wav_vec = wav_vec.squeeze(0)
        return wav_vec

    def rms(self, x):
        val = torch.sqrt(torch.mean(x ** 2))
        if val==0:
            val=1
        return val

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + v_context):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def __len__(self):
        return len(self.all_videos)
        # return 3000

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            wavpath = join(vidname, "audio.wav")
            img_names = natsorted(list(glob(join(vidname, '*.jpg'))), key=lambda y: y.lower())
            interval_st, interval_end = 0, len(img_names)
            if interval_end - interval_st <= tot_num_frames:
                continue
            pos_frame_id = random.randint(interval_st, interval_end - v_context)
            pos_wav = self.get_wav(wavpath, pos_frame_id)
            rms_pos_wav = self.rms(pos_wav)

            img_name = os.path.join(vidname, str(pos_frame_id)+'.jpg')
            window_fnames = self.get_window(img_name)

            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                # Image: e.g. (3, 109, 85) It's not a fit shape image.
                img = torchvision.io.read_image(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    # Shape (3, 96, 96)
                    img = self.cvresize(img)
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read:
                continue
            if random.choice([True, False]):
                y = torch.ones(1).float()
                wav = pos_wav
            else:
                y = torch.zeros(1).float()
                try_counter = 0
                while True:
                    neg_frame_id = random.randint(interval_st, interval_end - v_context)
                    if neg_frame_id != pos_frame_id:
                        wav = self.get_wav(wavpath, neg_frame_id)
                        if rms_pos_wav > 0.01:
                            break
                        else:
                            if self.rms(wav) > 0.01 or try_counter>10:
                                break
                        try_counter += 1

                if try_counter > 10:
                    continue
            aud_tensor = wav.type(torch.FloatTensor)

            # H, W, T, 3 --> T*3
            vid = torch.cat(window, 0) # (15, 96, 96)
            vid = vid[:, 48:].type(torch.FloatTensor) # (15, 48, 96)

            if torch.any(torch.isnan(vid)) or torch.any(torch.isnan(aud_tensor)):
                continue
            if vid==None or aud_tensor==None:
                continue

            return vid, aud_tensor, y

def train(device, teacher_model, student_model, 
        train_data_loader, test_data_loader,
        g_step, g_epoch, 
        optimizer, train_disloss, logloss,
        experiment_dir=None, checkpoint_interval=None, nepochs=None):
    
    global best_average_f1
    global global_step, global_epoch
    global_step = g_step
    global_epoch = g_epoch
    best_average_f1 = 0.0

    teacher_model.eval()

    resumed_step = global_step
    while global_epoch < nepochs:
        logger.info('-'*24 + f'Epoch {global_epoch+1}' + '-'*24)
        f1_scores = []
        running_loss = 0.
        running_logit_loss = 0.
        run_av_con_loss = 0.

        now_lr = optimizer.param_groups[0]['lr']
        logger.info(f"now learning rate is: {now_lr}")
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (vid, aud, y) in prog_bar:
            gt_aud = aud.to(device)
            vid = vid.to(device)
            y = y.to(device)
            mels = audio_feature_extractor(hparams, gt_aud)
            
            student_model.train()

            # produce loss label
            with torch.no_grad():
                soft_y, tea_fea_list = teacher_model(vid.clone().detach(), mels.clone().detach())

            out, stu_fea_list = student_model(vid.clone().detach(), mels.clone().detach())
            out_dict = {"tea_fea": tea_fea_list, "stu_fea": stu_fea_list, "y_t": soft_y, "y_s": out, "label": y} 

            ### Loss 
            loss = logloss(out, y.squeeze(-1).to(device))
            if args.loss_function != None:
                av_con_loss = train_disloss(out_dict)
                # print(f"av_con_loss: {av_con_loss}")
                total_loss = loss + av_con_loss
            else:
                total_loss = loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step(global_epoch+1)

            est_label = (out > 0.5).float()
            f1_metric = f1_score(y.clone().detach().cpu().numpy(), est_label.clone().detach().cpu().numpy(), average="weighted")

            f1_scores.append(f1_metric.item())
            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += total_loss.item()
            run_av_con_loss += av_con_loss.item()
            running_logit_loss += loss.item()

        f1_epoch = sum(f1_scores) / len(f1_scores)
        writer.add_scalars('f1_epoch', {'train': f1_epoch}, global_epoch)
        writer.add_scalars('loss_epoch', {'train': running_loss / (step + 1)}, global_epoch)
        logger.info(f"Epoch: [{global_epoch + 1}/{nepochs}]\t " +
                    f"Train loss: {(running_loss / (step + 1)):.5f}, " + 
                    f"kd_loss: {(run_av_con_loss / (step + 1)):.5f}," + 
                    f"loggit Loss: {(running_logit_loss / (step + 1)):.5f}, " + 
                    f"F1 score: {f1_epoch:.5f}")

        with torch.no_grad():
            true_better, best_average_f1 = eval_model(test_data_loader, device, teacher_model, student_model, experiment_dir, best_average_f1, nepochs)
            save_checkpoint(logger, hparams, student_model, optimizer, lr_scheduler, global_step, experiment_dir, global_epoch, nepochs, better=False)
    
            if true_better:
                save_checkpoint(logger, hparams, student_model, optimizer, lr_scheduler, global_step, experiment_dir, global_epoch, nepochs, better=true_better)

        global_epoch += 1

    logger.info("Finished training now!")

def eval_model(test_data_loader, device, teacher_model, model, experiment_dir, best_average_f1, nepochs=None):
    losses = []
    fitnet_losses = []
    running_loss=0
    running_floss=0
    better = False
    f1_scores = []
    prog_bar = tqdm(enumerate(test_data_loader))

    for step, (vid, aud, y) in prog_bar:
        model.eval()
        with torch.no_grad():
            gt_aud, vid, y = aud.to(device), vid.to(device), y.to(device)
            mels = audio_feature_extractor(hparams, gt_aud)
            #Teacher
            soft_y, tea_fea_list = teacher_model(vid.clone().detach(), mels.clone().detach())

            out, stu_fea_list = model(vid.clone().detach(), mels.clone().detach())
            out_dict = {"tea_fea": tea_fea_list, "stu_fea": stu_fea_list, "y_t": soft_y, "y_s": out, "label": y} 

            loss = logloss(out, y.squeeze(-1))
            floss = fitnet_loss(out_dict)
            losses.append(loss.item())
            fitnet_losses.append(floss.item())

            est_label = (out > 0.5).float()
            f1_metric = f1_score(y.clone().detach().cpu().numpy(), est_label.clone().detach().cpu().numpy(), average="weighted")
            f1_scores.append(f1_metric.item())
            running_loss += loss.item()
            running_floss += floss.item()

    averaged_loss = sum(losses) / len(losses)
    averaged_fitnet_loss = sum(fitnet_losses) / len(fitnet_losses)
    averaged_f1_score = sum(f1_scores) / len(f1_scores)
    writer.add_scalars('loss_epoch', {'val': averaged_loss}, global_epoch)
    writer.add_scalars('f1_epoch', {'val': averaged_f1_score}, global_epoch)

    logger.info(f"Epoch: [{global_epoch + 1}/{nepochs}]\t Val loss: {(running_loss / (step + 1)):.5f}, fitnetloss: {(running_floss / (step + 1)):.5f}, F1 score: {averaged_f1_score:.5f}, Best F1: {best_average_f1:.5f}")

    if averaged_f1_score > best_average_f1:
        best_average_f1 = averaged_f1_score
        better = True

    return better, best_average_f1

if __name__ == "__main__":
    args = parse_option()

    if args.note is not None:
        args.experiment_dir = args.experiment_dir + '_' + args.note

    writer = SummaryWriter(log_dir=os.path.join(args.experiment_dir, 'tensorboard'))
    logger = get_logger(os.path.join(args.experiment_dir, 'train.log'))

    global_step, global_epoch = 0, 0
    num_audio_elements = 3200  # 6400  # 16000/25 * syncnet_T
    tot_num_frames = 25  # buffer
    v_context = 5  # 10  # 5

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    logger.info('Use_cuda: {}'.format(use_cuda))
    
    #########
    experiment_dir = args.experiment_dir
    teacher_checkpoint_path, student_checkpoint_path = args.teacher_checkpoint, args.student_checkpoint

    if not os.path.exists(experiment_dir): os.mkdir(experiment_dir)
    logger.info(f'Model save path: {experiment_dir}')

    train_dataset, test_dataset = Dataset('pretrain'), Dataset('val')

    train_data_loader = data_utils.DataLoader(train_dataset, 
                                            batch_size=args.batch_size, 
                                            shuffle=True, 
                                            num_workers=args.num_thread,
                                            prefetch_factor=5, 
                                            pin_memory=True)

    test_data_loader = data_utils.DataLoader(test_dataset, 
                                            batch_size=args.batch_size, 
                                            num_workers=args.num_thread,
                                            prefetch_factor=5,
                                            pin_memory=True)

    logger.info(f"Load data, batch size is: {args.batch_size}.")

    # Model
    teacher_model = Tea_SyncTransformer().to(device)
    student_model = Stu_SyncTransformer(d_model=200).to(device)
    student_model_size = sum(p.numel() for p in student_model.parameters() if p.requires_grad) / 1e6
    logger.info('Total trainable params {:.2f} M'.format(student_model_size))

    # Opimizer
    optimizer = optim.Adam([p for p in student_model.parameters() if p.requires_grad], lr=args.learning_rate)
    logger.info(f"Init optimizer, and the learning rate is: {args.learning_rate}")

    scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler_steplr)

    ## Config
    config = ConfigParser()
    config.optionxform = str
    config.read(args.loss_config)
    cfg_dict = config._sections

    logloss = nn.BCEWithLogitsLoss()
    fitnet_loss = FitNetLoss(emb='last')
    if args.loss_function != None:
        try:
            print(f"loss_function: {args.loss_function}, loss_subcfg: {args.loss_subcfg}")
            loss_cfg_dict = cfg_dict[args.loss_subcfg]
            logger.info(loss_cfg_dict)
            train_disloss = importlib.import_module(f"distiller_zoo.{args.loss_function}").__getattribute__("Loss")(**loss_cfg_dict)
        except:
            logger.info('Loss config is None!')
            train_disloss = importlib.import_module(f"distiller_zoo.{args.loss_function}").__getattribute__("Loss")()
        logger.info(f"Define logit loss and Distillation Function: {args.loss_function}")

    if teacher_checkpoint_path is not None:
        logger.info(f'Load Teacher Model!')
        teacher_model, _, _ = load_checkpoint(logger, teacher_checkpoint_path, teacher_model, optimizer, lr_scheduler, scheduler_steplr, use_cuda, reset_optimizer=True)

    if student_checkpoint_path is not None:
        logger.info(f'resume student Model')
        student_model, g_step, g_epoch = load_checkpoint(logger, student_checkpoint_path, student_model, optimizer, scheduler_steplr, lr_scheduler, use_cuda, reset_optimizer=False)
        global_step, global_epoch = g_step, g_epoch

    train(device, teacher_model, student_model, 
        train_data_loader, test_data_loader, 
        global_step, global_epoch,
        optimizer, train_disloss, logloss,
        experiment_dir=experiment_dir,
        checkpoint_interval=100,
        nepochs=args.epoch)
