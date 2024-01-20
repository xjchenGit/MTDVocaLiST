from regex import P
import torch
import torch.nn as nn
import torch.nn.functional as F
# from geomloss import SamplesLoss

class Loss(nn.Module):
    """ Distilling the Knowledge using MiniLM."""
    def __init__(self, state1, layer1, state2, layer2, state3, layer3, temperature, alpha):
        super(Loss, self).__init__()
        self.state_list = [state1, state2, state3]
        self.layer_list = [int(layer1) - 1, int(layer2) - 1, int(layer3) - 1]
        # self.state1 = state1
        # self.layer1 = int(layer1) - 1
        # self.state2 = state2
        # self.layer2 = int(layer2) - 1
        # self.state3 = state3
        # self.layer3 = int(layer3) - 1

        self.Tem = int(temperature)
        self.alpha = float(alpha)

        self.kl_div = nn.KLDivLoss(reduction="mean")

    def _softmax_w_temperature(self, m, temperature=1):
        return F.softmax(m.float() / temperature, dim=-1).type_as(m)

        # self.feature_list['AV_Trans'] = {'av_emb': av_embedding, 'va_emb': va_embedding, 'av_qk': av_qk, 'va_qk': va_qk, 
        #                                 'av_q_norm': av_q_norm, 'va_q_norm': va_q_norm, 'av_k': av_k, 'va_k': va_k,
        #                                 'av_v_norm': av_v_norm, 'va_v_norm': va_v_norm, 'av_v': av_v, 'va_v': va_v}

    def _preprocess_qkv_layer_select1(self, state, layer, fea_dict, w_softmax=False):
        stu_fea, tea_fea = fea_dict['stu_fea'], fea_dict['tea_fea']
        
        if state == 'fus':
            stu_q_norm = stu_fea['Fus_Trans']['mem_q_norm'][layer]
            tea_q_norm = tea_fea['Fus_Trans']['mem_q_norm'][layer]
            stu_q = stu_fea['Fus_Trans']['mem_q'][layer]
            tea_q = tea_fea['Fus_Trans']['mem_q'][layer]
        elif state == 'av':
            stu_q_norm = stu_fea['AV_Trans']['av_q_norm'][layer]
            tea_q_norm = tea_fea['AV_Trans']['av_q_norm'][layer]
            stu_q = stu_fea['AV_Trans']['av_q'][layer]
            tea_q = tea_fea['AV_Trans']['av_q'][layer]
        elif state == 'va':
            stu_q_norm = stu_fea['AV_Trans']['va_q_norm'][layer]
            tea_q_norm = tea_fea['AV_Trans']['va_q_norm'][layer]
            stu_q = stu_fea['AV_Trans']['va_q'][layer]
            tea_q = tea_fea['AV_Trans']['va_q'][layer]

        stu_qq = torch.mm(stu_q_norm, stu_q.transpose(0, 1))
        tea_qq = torch.mm(tea_q_norm, tea_q.transpose(0, 1))

        if state == 'fus':
            stu_k_norm = stu_fea['Fus_Trans']['mem_k_norm'][layer]
            tea_k_norm = tea_fea['Fus_Trans']['mem_k_norm'][layer]
            stu_k = stu_fea['Fus_Trans']['mem_k'][layer]
            tea_k = tea_fea['Fus_Trans']['mem_k'][layer]
        elif state == 'av':
            stu_k_norm = stu_fea['AV_Trans']['av_k_norm'][layer]
            tea_k_norm = tea_fea['AV_Trans']['av_k_norm'][layer]
            stu_k = stu_fea['AV_Trans']['av_k'][layer]
            tea_k = tea_fea['AV_Trans']['av_k'][layer]
        elif state == 'va':
            stu_k_norm = stu_fea['AV_Trans']['va_k_norm'][layer]
            tea_k_norm = tea_fea['AV_Trans']['va_k_norm'][layer]
            stu_k = stu_fea['AV_Trans']['va_k'][layer]
            tea_k = tea_fea['AV_Trans']['va_k'][layer]

        stu_kk = torch.mm(stu_k_norm, stu_k.transpose(0, 1))
        tea_kk = torch.mm(tea_k_norm, tea_k.transpose(0, 1))

        if state == 'fus':
            stu_v_norm = stu_fea['Fus_Trans']['mem_v_norm'][layer]
            tea_v_norm = tea_fea['Fus_Trans']['mem_v_norm'][layer]
            stu_v = stu_fea['Fus_Trans']['mem_v'][layer]
            tea_v = tea_fea['Fus_Trans']['mem_v'][layer]
        elif state == 'av':
            stu_v_norm = stu_fea['AV_Trans']['av_v_norm'][layer]
            tea_v_norm = tea_fea['AV_Trans']['av_v_norm'][layer]
            stu_v = stu_fea['AV_Trans']['av_v'][layer]
            tea_v = tea_fea['AV_Trans']['av_v'][layer]
        elif state == 'va':
            stu_v_norm = stu_fea['AV_Trans']['va_v_norm'][layer]
            tea_v_norm = tea_fea['AV_Trans']['va_v_norm'][layer]
            stu_v = stu_fea['AV_Trans']['va_v'][layer]
            tea_v = tea_fea['AV_Trans']['va_v'][layer]

        stu_vv = torch.mm(stu_v_norm, stu_v.transpose(0, 1))
        tea_vv = torch.mm(tea_v_norm, tea_v.transpose(0, 1))

        sm_stu_qq = self._softmax_w_temperature(stu_qq, self.Tem)
        sm_tea_qq = self._softmax_w_temperature(tea_qq, self.Tem)
        
        sm_stu_kk = self._softmax_w_temperature(stu_kk, self.Tem)
        sm_tea_kk = self._softmax_w_temperature(tea_kk, self.Tem)

        sm_stu_vv = self._softmax_w_temperature(stu_vv, self.Tem)
        sm_tea_vv = self._softmax_w_temperature(tea_vv, self.Tem)

        if w_softmax:
            return sm_tea_qq, sm_stu_qq, sm_tea_kk, sm_stu_kk, sm_tea_vv, sm_stu_vv
        else:
            return tea_qq, stu_qq, tea_kk, stu_kk, tea_vv, stu_vv

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
        losses = 0.
        len_state = len(self.state_list)
        for i in range(len_state):
            sta = self.state_list[i]
            lay = self.layer_list[i]
            tea_qq, stu_qq, tea_kk, stu_kk, tea_vv, stu_vv = self._preprocess_qkv_layer_select1(state=sta, layer=lay, fea_dict=anchor, w_softmax=True)
            loss = self.Tem**2 * self.kl_div(stu_qq.log(), tea_qq) + \
                self.Tem**2 * self.kl_div(stu_kk.log(), tea_kk) + \
                self.Tem**2 * self.kl_div(stu_vv.log(), tea_vv)
                
            losses += loss
        
        # tea_fus_qq1, stu_fus_qq1, tea_fus_kk1, stu_fus_kk1, tea_fus_vv1, stu_fus_vv1 = self._preprocess_qkv_layer_select1(state=, anchor, w_softmax=True)
        # tea_fus_qq2, stu_fus_qq2, tea_fus_kk2, stu_fus_kk2, tea_fus_vv2, stu_fus_vv2 = self._preprocess_qkv_layer_select2(anchor, w_softmax=True)
        # tea_fus_qq3, stu_fus_qq3, tea_fus_kk3, stu_fus_kk3, tea_fus_vv3, stu_fus_vv3 = self._preprocess_qkv_layer_select3(anchor, w_softmax=True)

        # loss1 = self.Tem**2 * self.kl_div(stu_fus_qq1.log(), tea_fus_qq1) + \
        #         self.Tem**2 * self.kl_div(stu_fus_kk1.log(), tea_fus_kk1) + \
        #         self.Tem**2 * self.kl_div(stu_fus_vv1.log(), tea_fus_vv1)
                
        # loss2 = self.Tem**2 * self.kl_div(stu_fus_qq2.log(), tea_fus_qq2) + \
        #         self.Tem**2 * self.kl_div(stu_fus_kk2.log(), tea_fus_kk2) + \
        #         self.Tem**2 * self.kl_div(stu_fus_vv2.log(), tea_fus_vv2)
                
        # loss3 = self.Tem**2 * self.kl_div(stu_fus_qq3.log(), tea_fus_qq3) + \
        #         self.Tem**2 * self.kl_div(stu_fus_kk3.log(), tea_fus_kk3) + \
        #         self.Tem**2 * self.kl_div(stu_fus_vv3.log(), tea_fus_vv3)
        
        # return self.alpha * (loss1 + loss2 + loss3)
        return self.alpha * losses




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
