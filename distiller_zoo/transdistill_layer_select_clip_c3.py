from regex import P
import torch
import torch.nn as nn
import torch.nn.functional as F
# from geomloss import SamplesLoss

class Loss(nn.Module):
    """ Distilling the Knowledge using MiniLM."""
    def __init__(self, state1, layer1, state2, layer2, state3, layer3, temperature, alpha):
        super(Loss, self).__init__()
        self.state1 = state1
        self.layer1 = int(layer1) - 1
        self.state2 = state2
        self.layer2 = int(layer2) - 1
        self.state3 = state3
        self.layer3 = int(layer3) - 1

        self.Tem = int(temperature)
        self.alpha = float(alpha)

        self.kl_div = nn.KLDivLoss(reduction="mean")

    def _softmax_w_temperature(self, m, temperature=1):
        return F.softmax(m.float() / temperature, dim=-1).type_as(m)

        # self.feature_list['AV_Trans'] = {'av_emb': av_embedding, 'va_emb': va_embedding, 'av_qk': av_qk, 'va_qk': va_qk, 
        #                                 'av_q_norm': av_q_norm, 'va_q_norm': va_q_norm, 'av_k': av_k, 'va_k': va_k,
        #                                 'av_v_norm': av_v_norm, 'va_v_norm': va_v_norm, 'av_v': av_v, 'va_v': va_v}

    def _preprocess_qkv_layer_select1(self, fea_dict, w_softmax=False):
        stu_fea, tea_fea = fea_dict['stu_fea'], fea_dict['tea_fea']

        stu_fus_qk = stu_fea['Fus_Trans']['mem_qk'][self.layer1]
        tea_fus_qk = tea_fea['Fus_Trans']['mem_qk'][self.layer1]

        if self.state1 == 'fus':
            stu_fus_qk = stu_fea['Fus_Trans']['mem_qk'][self.layer1]
            tea_fus_qk = tea_fea['Fus_Trans']['mem_qk'][self.layer1]
        elif self.state1 == 'av':
            stu_fus_qk = stu_fea['AV_Trans']['av_qk'][self.layer1] 
            tea_fus_qk = tea_fea['AV_Trans']['av_qk'][self.layer1]
        elif self.state1 == 'va':
            stu_fus_qk = stu_fea['AV_Trans']['va_qk'][self.layer1].clamp(min=1e-5)
            tea_fus_qk = tea_fea['AV_Trans']['va_qk'][self.layer1].clamp(min=1e-5)

        sm_stu_fus_qk = self._softmax_w_temperature(stu_fus_qk, self.Tem)
        sm_tea_fus_qk = self._softmax_w_temperature(tea_fus_qk, self.Tem)

        if self.state1 == 'fus':
            stu_fus_v_norm = stu_fea['Fus_Trans']['mem_v_norm'][self.layer1]
            tea_fus_v_norm = tea_fea['Fus_Trans']['mem_v_norm'][self.layer1]
            stu_fus_v = stu_fea['Fus_Trans']['men_v'][self.layer1]
            tea_fus_v = tea_fea['Fus_Trans']['men_v'][self.layer1]
        elif self.state1 == 'av':
            stu_fus_v_norm = stu_fea['AV_Trans']['av_v_norm'][self.layer1]
            tea_fus_v_norm = tea_fea['AV_Trans']['av_v_norm'][self.layer1]
            stu_fus_v = stu_fea['AV_Trans']['av_v'][self.layer1]
            tea_fus_v = tea_fea['AV_Trans']['av_v'][self.layer1]
        elif self.state1 == 'va':
            stu_fus_v_norm = stu_fea['AV_Trans']['va_v_norm'][self.layer1]
            tea_fus_v_norm = tea_fea['AV_Trans']['va_v_norm'][self.layer1]
            stu_fus_v = stu_fea['AV_Trans']['va_v'][self.layer1]
            tea_fus_v = tea_fea['AV_Trans']['va_v'][self.layer1]

        stu_fus_vv = torch.bmm(stu_fus_v_norm, stu_fus_v.transpose(1, 2))
        tea_fus_vv = torch.bmm(tea_fus_v_norm, tea_fus_v.transpose(1, 2))

        sm_stu_fus_vv = self._softmax_w_temperature(stu_fus_vv, self.Tem)
        sm_tea_fus_vv = self._softmax_w_temperature(tea_fus_vv, self.Tem)

        if w_softmax:
            return sm_tea_fus_qk, sm_stu_fus_qk, sm_tea_fus_vv, sm_stu_fus_vv
        else:
            return tea_fus_qk, stu_fus_qk, tea_fus_vv, stu_fus_vv

    def _preprocess_qkv_layer_select2(self, fea_dict, w_softmax=False):
        stu_fea, tea_fea = fea_dict['stu_fea'], fea_dict['tea_fea']

        stu_fus_qk = stu_fea['Fus_Trans']['mem_qk'][self.layer2]
        tea_fus_qk = tea_fea['Fus_Trans']['mem_qk'][self.layer2]

        if self.state2 == 'fus':
            stu_fus_qk = stu_fea['Fus_Trans']['mem_qk'][self.layer2]
            tea_fus_qk = tea_fea['Fus_Trans']['mem_qk'][self.layer2]
        elif self.state2 == 'av':
            stu_fus_qk = stu_fea['AV_Trans']['av_qk'][self.layer2] 
            tea_fus_qk = tea_fea['AV_Trans']['av_qk'][self.layer2]
        elif self.state2 == 'va':
            stu_fus_qk = stu_fea['AV_Trans']['va_qk'][self.layer2].clamp(min=1e-5)
            tea_fus_qk = tea_fea['AV_Trans']['va_qk'][self.layer2].clamp(min=1e-5)

        sm_stu_fus_qk = self._softmax_w_temperature(stu_fus_qk, self.Tem)
        sm_tea_fus_qk = self._softmax_w_temperature(tea_fus_qk, self.Tem)

        if self.state2 == 'fus':
            stu_fus_v_norm = stu_fea['Fus_Trans']['mem_v_norm'][self.layer2]
            tea_fus_v_norm = tea_fea['Fus_Trans']['mem_v_norm'][self.layer2]
            stu_fus_v = stu_fea['Fus_Trans']['men_v'][self.layer2]
            tea_fus_v = tea_fea['Fus_Trans']['men_v'][self.layer2]
        elif self.state2 == 'av':
            stu_fus_v_norm = stu_fea['AV_Trans']['av_v_norm'][self.layer2]
            tea_fus_v_norm = tea_fea['AV_Trans']['av_v_norm'][self.layer2]
            stu_fus_v = stu_fea['AV_Trans']['av_v'][self.layer2]
            tea_fus_v = tea_fea['AV_Trans']['av_v'][self.layer2]
        elif self.state2 == 'va':
            stu_fus_v_norm = stu_fea['AV_Trans']['va_v_norm'][self.layer2]
            tea_fus_v_norm = tea_fea['AV_Trans']['va_v_norm'][self.layer2]
            stu_fus_v = stu_fea['AV_Trans']['va_v'][self.layer2]
            tea_fus_v = tea_fea['AV_Trans']['va_v'][self.layer2]

        stu_fus_vv = torch.bmm(stu_fus_v_norm, stu_fus_v.transpose(1, 2))
        tea_fus_vv = torch.bmm(tea_fus_v_norm, tea_fus_v.transpose(1, 2))

        sm_stu_fus_vv = self._softmax_w_temperature(stu_fus_vv, self.Tem)
        sm_tea_fus_vv = self._softmax_w_temperature(tea_fus_vv, self.Tem)

        if w_softmax:
            return sm_tea_fus_qk, sm_stu_fus_qk, sm_tea_fus_vv, sm_stu_fus_vv
        else:
            return tea_fus_qk, stu_fus_qk, tea_fus_vv, stu_fus_vv

    def _preprocess_qkv_layer_select3(self, fea_dict, w_softmax=False):
        stu_fea, tea_fea = fea_dict['stu_fea'], fea_dict['tea_fea']

        stu_fus_qk = stu_fea['Fus_Trans']['mem_qk'][self.layer3]
        tea_fus_qk = tea_fea['Fus_Trans']['mem_qk'][self.layer3]

        if self.state3 == 'fus':
            stu_fus_qk = stu_fea['Fus_Trans']['mem_qk'][self.layer3]
            tea_fus_qk = tea_fea['Fus_Trans']['mem_qk'][self.layer3]
        elif self.state3 == 'av':
            stu_fus_qk = stu_fea['AV_Trans']['av_qk'][self.layer3] 
            tea_fus_qk = tea_fea['AV_Trans']['av_qk'][self.layer3]
        elif self.state3 == 'va':
            stu_fus_qk = stu_fea['AV_Trans']['va_qk'][self.layer3].clamp(min=1e-5)
            tea_fus_qk = tea_fea['AV_Trans']['va_qk'][self.layer3].clamp(min=1e-5)

        sm_stu_fus_qk = self._softmax_w_temperature(stu_fus_qk, self.Tem)
        sm_tea_fus_qk = self._softmax_w_temperature(tea_fus_qk, self.Tem)

        if self.state3 == 'fus':
            stu_fus_v_norm = stu_fea['Fus_Trans']['mem_v_norm'][self.layer3]
            tea_fus_v_norm = tea_fea['Fus_Trans']['mem_v_norm'][self.layer3]
            stu_fus_v = stu_fea['Fus_Trans']['men_v'][self.layer3]
            tea_fus_v = tea_fea['Fus_Trans']['men_v'][self.layer3]
        elif self.state3 == 'av':
            stu_fus_v_norm = stu_fea['AV_Trans']['av_v_norm'][self.layer3]
            tea_fus_v_norm = tea_fea['AV_Trans']['av_v_norm'][self.layer3]
            stu_fus_v = stu_fea['AV_Trans']['av_v'][self.layer3]
            tea_fus_v = tea_fea['AV_Trans']['av_v'][self.layer3]
        elif self.state3 == 'va':
            stu_fus_v_norm = stu_fea['AV_Trans']['va_v_norm'][self.layer3]
            tea_fus_v_norm = tea_fea['AV_Trans']['va_v_norm'][self.layer3]
            stu_fus_v = stu_fea['AV_Trans']['va_v'][self.layer3]
            tea_fus_v = tea_fea['AV_Trans']['va_v'][self.layer3]

        stu_fus_vv = torch.bmm(stu_fus_v_norm, stu_fus_v.transpose(1, 2))
        tea_fus_vv = torch.bmm(tea_fus_v_norm, tea_fus_v.transpose(1, 2))

        sm_stu_fus_vv = self._softmax_w_temperature(stu_fus_vv, self.Tem)
        sm_tea_fus_vv = self._softmax_w_temperature(tea_fus_vv, self.Tem)

        if w_softmax:
            return sm_tea_fus_qk, sm_stu_fus_qk, sm_tea_fus_vv, sm_stu_fus_vv
        else:
            return tea_fus_qk, stu_fus_qk, tea_fus_vv, stu_fus_vv

    def forward(self, anchor):

        tea_fus_qk1, stu_fus_qk1, tea_fus_vv1, stu_fus_vv1 = self._preprocess_qkv_layer_select1(anchor, w_softmax=True)
        tea_fus_qk2, stu_fus_qk2, tea_fus_vv2, stu_fus_vv2 = self._preprocess_qkv_layer_select2(anchor, w_softmax=True)
        tea_fus_qk3, stu_fus_qk3, tea_fus_vv3, stu_fus_vv3 = self._preprocess_qkv_layer_select3(anchor, w_softmax=True)

        loss1 =  self.Tem**2 * self.kl_div(stu_fus_qk1.log(), tea_fus_qk1) + \
                    self.Tem**2 * self.kl_div(stu_fus_vv1.log(), tea_fus_vv1)

        loss2 =  self.Tem**2 * self.kl_div(stu_fus_qk2.log(), tea_fus_qk2) + \
                   self.Tem**2 *  self.kl_div(stu_fus_vv2.log(), tea_fus_vv2)

        loss3 =  self.Tem**2 * self.kl_div(stu_fus_qk3.log(), tea_fus_qk3) + \
                  self.Tem**2 * self.kl_div(stu_fus_vv3.log(), tea_fus_vv3)
        
        return self.alpha * (loss1 + loss2 + loss3)




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

    sad_kl_Loss = Loss(state='all')
    res_av_con_loss = sad_kl_Loss(input_block)
    print(f"res_av_con_loss: {res_av_con_loss}")

    res_av_con_loss = sad_kl_Loss(input_block)
    print(f"res_av_con_loss: {res_av_con_loss}")
