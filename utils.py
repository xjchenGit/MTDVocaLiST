import logging
import torch
from torchaudio.transforms import MelScale

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # Output to file
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Output to terminal
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def _load(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(logger, path, model, optimizer, scheduler, use_cuda, scheduler_steplr, reset_optimizer=False):
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, use_cuda)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            logger.info("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
            try: 
                scheduler.load_state_dict(checkpoint["scheduler"])
            except:
                sch_state = {'multiplier': 1, 'total_epoch': 10, 
                            'after_scheduler': scheduler_steplr, 
                            'finished': False, 'base_lrs': [5e-05], 'last_epoch': 98, 
                            '_step_count': 73794, 'verbose': False, '_get_lr_called_within_step': False, 
                            '_last_lr': [2.0480000000000004e-05]}
                scheduler.load_state_dict(sch_state)
            # print(f"checkpoint scheduler: {checkpoint['scheduler']}")
            """
            {'multiplier': 1, 'total_epoch': 10, 'after_scheduler': <torch.optim.lr_scheduler.StepLR object at 0x7f68f84501c0>, 
            'finished': False, 'base_lrs': [5e-05], 'last_epoch': 98, '_step_count': 73794, 'verbose': False, '_get_lr_called_within_step': False, 
            '_last_lr': [2.0480000000000004e-05]}
            """

    return model, checkpoint["global_step"], checkpoint["global_epoch"]

def audio_feature_extractor(hparams, audio):
    melscale = MelScale(n_mels=hparams.num_mels, sample_rate=hparams.sample_rate, f_min=hparams.fmin, 
                        f_max=hparams.fmax, n_stft=hparams.n_stft, norm='slaney', mel_scale='slaney').to(0)

    spec = torch.stft(audio, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size,
                        window=torch.hann_window(hparams.win_size).to(audio.device), return_complex=True)
    melspec = melscale(torch.abs(spec.detach().clone()).float())
    melspec_tr1 = (20 * torch.log10(torch.clamp(melspec, min=hparams.MIN_LEVEL))) - hparams.ref_level_db
    #NORMALIZED MEL
    normalized_mel = torch.clip((2 * hparams.max_abs_value) * ((melspec_tr1 + hparams.TOP_DB) / hparams.TOP_DB) - hparams.max_abs_value,
                                -hparams.max_abs_value, hparams.max_abs_value)
    mels = normalized_mel[:, :, :-1].unsqueeze(1)
    return mels