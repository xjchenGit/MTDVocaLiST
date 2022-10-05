import torch.nn as nn

class Loss(nn.Module):
    """ Fitnets: hints for thin deep nets, ICLR 2015 """
    def __init__(self, fuslayer, avlayer, valayer, tea_emb, stu_emb, Beta):
        super(Loss, self).__init__()
        self.fuslayer = int(fuslayer)
        self.avlayer = int(avlayer)
        self.valayer = int(valayer)

        self.linear = nn.Linear(int(tea_emb), int(stu_emb)).cuda()
        self.crit = nn.MSELoss()
        self.Beta = float(Beta)

    def forward(self, out_dict):
        fea_s, fea_t = out_dict['stu_fea'], out_dict['tea_fea']

        f_t = fea_t['Fus_Trans']['mem_emb'][self.fuslayer]
        f_s = fea_s['Fus_Trans']['mem_emb'][self.fuslayer]
        # print(f"f_t: {f_t.shape}, f_s: {f_s.shape}")
        f_t = self.linear(f_t)

        f_t_av = self.linear(fea_t['AV_Trans']['av_emb'][self.avlayer])
        f_s_av = fea_s['AV_Trans']['av_emb'][self.avlayer]

        f_t_va = self.linear(fea_t['AV_Trans']['va_emb'][self.valayer])
        f_s_va = fea_s['AV_Trans']['va_emb'][self.valayer]
        loss = self.crit(f_s_av, f_t_av) + self.crit(f_s_va, f_t_va) + self.crit(f_s, f_t)

        return self.Beta * loss