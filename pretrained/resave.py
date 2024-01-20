import torch
# from models import student_v2

# checkpoint_path = "Best.pth"
# cpk = torch.load(checkpoint_path, map_location='cpu')

# # model: dict_keys(['state_dict', 'aw_loss', 'optimizer', 'global_step', 'global_epoch', 'scheduler', 'lr'])
# parameter = {key: value for key, value in cpk.items() if key in ['state_dict']}
# torch.save(parameter, "pure_MTDVocaLiST.pth")

### repro Vocalist.pth
checkpoint_path = "repro_Vocalist.pth"
cpk = torch.load(checkpoint_path, map_location='cpu')

# model: dict_keys(['state_dict', 'aw_loss', 'optimizer', 'global_step', 'global_epoch', 'scheduler', 'lr'])
parameter = {key: value for key, value in cpk.items() if key in ['state_dict']}
torch.save(parameter, "pure_Vocalist.pth")
