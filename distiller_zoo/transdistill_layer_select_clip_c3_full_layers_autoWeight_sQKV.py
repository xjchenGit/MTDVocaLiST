from regex import P
import torch
import torch.nn as nn
import torch.nn.functional as F
# from geomloss import SamplesLoss

class Loss(nn.Module):
    """ Distilling the Knowledge using MiniLM."""
    def __init__(self, temperature, alpha):
        super(Loss, self).__init__()

        self.Tem = int(temperature)
        self.alpha = int(alpha)

        self.kl_div = nn.KLDivLoss(reduction="mean")
        
        # The self.params consist of 25 params.
        # + The first one is the prepare for the params of logit loss.
        # + The 
        p_num = self.alpha + 1
        params = torch.ones(p_num, requires_grad=True)
        self.params = nn.Parameter(params)
        # self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.nnSoftmax = nn.Softmax(dim=0)

    def _softmax_w_temperature(self, m, temperature=1):
        return F.softmax(m.float() / temperature, dim=-1).type_as(m)

    def _preprocess_qkv_layers(self, state, fea_dict, w_softmax=False):
        stu_fea, tea_fea = fea_dict['stu_fea'], fea_dict['tea_fea']

        if state == 'fus':
            stu_qk_list = stu_fea['Fus_Trans']['mem_qk']
            tea_qk_list = tea_fea['Fus_Trans']['mem_qk']
        elif state == 'av':
            stu_qk_list = stu_fea['AV_Trans']['av_qk'] 
            tea_qk_list = tea_fea['AV_Trans']['av_qk']
        elif state == 'va':
            stu_qk_list = [s.clamp(min=1e-5) for s in stu_fea['AV_Trans']['va_qk']]
            tea_qk_list = [t.clamp(min=1e-5) for t in tea_fea['AV_Trans']['va_qk']]

        if state == 'fus':
            stu_v_norm_list = stu_fea['Fus_Trans']['mem_v_norm']
            tea_v_norm_list = tea_fea['Fus_Trans']['mem_v_norm']
            stu_v_list = stu_fea['Fus_Trans']['men_v']
            tea_v_list = tea_fea['Fus_Trans']['men_v']
        elif state == 'av':
            stu_v_norm_list = stu_fea['AV_Trans']['av_v_norm']
            tea_v_norm_list = tea_fea['AV_Trans']['av_v_norm']
            stu_v_list = stu_fea['AV_Trans']['av_v']
            tea_v_list = tea_fea['AV_Trans']['av_v']
        elif state == 'va':
            stu_v_norm_list = stu_fea['AV_Trans']['va_v_norm']
            tea_v_norm_list = tea_fea['AV_Trans']['va_v_norm']
            stu_v_list = stu_fea['AV_Trans']['va_v']
            tea_v_list = tea_fea['AV_Trans']['va_v']
        
        stu_vv_list, tea_vv_list = [], []
        len_v_norm_list = len(tea_v_norm_list)
        
        for i in range(len_v_norm_list):
            tea_v_norm, stu_v_norm = tea_v_norm_list[i], stu_v_norm_list[i]
            tea_v, stu_v = tea_v_list[i], stu_v_list[i]
            tea_vv = torch.bmm(tea_v_norm, tea_v.transpose(1, 2))
            stu_vv = torch.bmm(stu_v_norm, stu_v.transpose(1, 2))
            stu_vv_list.append(stu_vv)
            tea_vv_list.append(tea_vv)

        if w_softmax:
            sm_stu_qk_list = [self._softmax_w_temperature(s, self.Tem) for s in stu_qk_list]
            sm_tea_qk_list = [self._softmax_w_temperature(t, self.Tem) for t in tea_qk_list]
            
            sm_stu_vv_list = [self._softmax_w_temperature(s, self.Tem) for s in stu_vv_list]
            sm_tea_vv_list = [self._softmax_w_temperature(t, self.Tem) for t in tea_vv_list]
        
            return sm_tea_qk_list, sm_stu_qk_list, sm_tea_vv_list, sm_stu_vv_list
        else:
            return tea_qk_list, stu_qk_list, tea_vv_list, stu_vv_list

    def forward(self, anchor, logit_loss):
        
        # self.params.clamp(min=1e-5, max=1e+5)
        # self.params = torch.sigmoid(self.params)
        # self.params.data = self.sigmoid(self.params.data)
        # self.params.data = self.params.data - self.params.data.max()
            
        sm_params = self.nnSoftmax(self.params)
        
        layer_type_list = ["fus", "av", "va"]
        
        # init the losses with logit loss multipy by the first params.
        losses = logit_loss * sm_params[0]
        
        for i, ty in enumerate(layer_type_list):
            tea_qk_list, stu_qk_list, tea_vv_list, stu_vv_list = self._preprocess_qkv_layers(state=ty, fea_dict=anchor, w_softmax=True)
            
            for j in range(len(tea_qk_list)):
                tea_qk, stu_qk, tea_vv, stu_vv = tea_qk_list[j], stu_qk_list[j], tea_vv_list[j], stu_vv_list[j]
                
                # The index of param
                qkv_idx = i * len(tea_qk_list) + j + 1
                
                loss = sm_params[qkv_idx] * (self.Tem**2 * self.kl_div(stu_qk.log(), tea_qk) + \
                                            self.Tem**2 * self.kl_div(stu_vv.log(), tea_vv))

                losses += loss
        
        kd_loss = losses - logit_loss * sm_params[0]       

        return losses, kd_loss, sm_params




if __name__ == "__main__":
    K = 128
    tea_dim = 512
    stu_dim = 256
    att_head = 8
    input_block = { 'tea_fea': {'Fus_Trans':{'mem_emb': torch.randn(16, K, tea_dim).cuda(), 
                                                'mem_qk': torch.randn(K * att_head, 16, 5).cuda(),
                                                'mem_q_norm': torch.randn(K * att_head, 16, 64).cuda(),
                                                'mem_k': torch.randn(K * att_head, 5, 64).cuda(),
                                                'mem_v_norm': torch.randn(K * att_head, 5, 64).cuda(),
                                                'men_v': torch.randn(K * att_head, 5, 64).cuda()}},
                    'stu_fea': {'Fus_Trans':{'mem_emb': torch.randn(16, K, stu_dim).cuda(), 
                                            'mem_qk': torch.randn(K * att_head, 16, 5).cuda(),
                                            'mem_q_norm': torch.randn(K * att_head, 16, 32).cuda(),
                                            'mem_k': torch.randn(K * att_head, 5, 32).cuda(),
                                            'mem_v_norm': torch.randn(K * att_head, 5, 32).cuda(),
                                            'men_v': torch.randn(K * att_head, 5, 32).cuda()}},
                    'y_s': torch.randn(K).cuda(),
                    'y_t': torch.randn(K).cuda(),
                    'label': torch.randn(K, 1).cuda()}
    K2 = 8176
    target_block = { 'tea_fea': {'Fus_Trans':{'mem_emb': torch.randn(16, K2, tea_dim).cuda(), 
                                                'mem_qk': torch.randn(K2 * att_head, 16, 5).cuda(),
                                                'mem_q_norm': torch.randn(K2 * att_head, 16, 64).cuda(),
                                                'mem_k': torch.randn(K2 * att_head, 5, 64).cuda(),
                                                'mem_v_norm': torch.randn(K2 * att_head, 5, 64).cuda(),
                                                'men_v': torch.randn(K2 * att_head, 5, 64).cuda()}},
                    'stu_fea': {'Fus_Trans':{'mem_emb': torch.randn(16, K2, stu_dim).cuda(), 
                                            'mem_qk': torch.randn(K2 * att_head, 16, 5).cuda(),
                                            'mem_q_norm': torch.randn(K2 * att_head, 16, 32).cuda(),
                                            'mem_k': torch.randn(K2 * att_head, 5, 32).cuda(),
                                            'mem_v_norm': torch.randn(K2 * att_head, 5, 32).cuda(),
                                            'men_v': torch.randn(K2 * att_head, 5, 32).cuda()}},
                    'y_s': torch.randn(K2).cuda(),
                    'y_t': torch.randn(K2).cuda(),
                    'label': torch.randn(K2, 1).cuda()}

    # sad_kl_Loss = Loss(cT=5, pT=5)
    # res_av_con_loss = sad_kl_Loss(input_block, target_block)
    # print(f"res_av_con_loss: {res_av_con_loss}")

    # res_av_con_loss = sad_kl_Loss(input_block, target_block, states="XBM")
    # print(f"res_av_con_loss: {res_av_con_loss}")

    sad_kl_Loss = Loss(state1='fus', 
                       layer1=4,
                       state2='av',
                       layer2=3, 
                       state3='va',
                       layer3=1,
                       temperature=1,
                       alpha=1)
    res_av_con_loss = sad_kl_Loss(input_block)
    print(f"res_av_con_loss: {res_av_con_loss}")

    # res_av_con_loss = sad_kl_Loss(input_block)
    # print(f"res_av_con_loss: {res_av_con_loss}")
