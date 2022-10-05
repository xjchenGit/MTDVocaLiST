from regex import P
import torch
import torch.nn as nn
import torch.nn.functional as F
# from geomloss import SamplesLoss

class Loss(nn.Module):
    """ Distilling the Knowledge using SADWII."""
    def __init__(self, cT, pT):
        super(Loss, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction="mean")
        self.cT = float(cT)
        self.pT = float(pT)

        print(f"self.cT: {self.cT}, self.pT: {self.pT}")

    def _softmax_w_temperature(self, m, temperature=1):
        return F.softmax(m.float() / temperature, dim=-1).type_as(m)

    def _log_softmax_w_temperature(self, m, temperature=1):
        return F.log_softmax(m.float() / temperature, dim=-1).type_as(m)

    def _pair_wise_sim_map(self, fea_0, fea_1, temp=1):
        print(f"fea_0: {fea_0.shape}, fea_1: {fea_1.shape}")
        sim_map_0_1 = torch.bmm(fea_0, fea_1.transpose(1, 2))
        sim_map_0_1 = self._softmax_w_temperature(sim_map_0_1, temp)
        return sim_map_0_1

        # self.feature_list['AV_Trans'] = {'av_emb': av_embedding, 'va_emb': va_embedding, 'av_qk': av_qk, 'va_qk': va_qk, 
        #                                 'av_q_norm': av_q_norm, 'va_q_norm': va_q_norm, 'av_k': av_k, 'va_k': va_k,
        #                                 'av_v_norm': av_v_norm, 'va_v_norm': va_v_norm, 'av_v': av_v, 'va_v': va_v}

    def _preprocess_cro_att_av_qkv(self, fea_dict, temperature, w_softmax=False):
        stu_fea, tea_fea = fea_dict['stu_fea'], fea_dict['tea_fea']

        stu_ca_qk = stu_fea['AV_Trans']['av_qk']
        tea_ca_qk = tea_fea['AV_Trans']['av_qk']

        # print(f"tea_ca_qk: {tea_ca_qk.shape}, stu_ca_qk: {stu_ca_qk.shape}")

        sm_stu_ca_qk = self._softmax_w_temperature(stu_ca_qk, temperature)
        sm_tea_ca_qk = self._softmax_w_temperature(tea_ca_qk, temperature)
        # print(f"sm_tea_ca_qk: {sm_tea_ca_qk[0][0].log()}, sm_stu_ca_qk: {sm_stu_ca_qk[0][0].log()}")

        stu_ca_v_norm = stu_fea['AV_Trans']['av_v_norm']
        tea_ca_v_norm = tea_fea['AV_Trans']['av_v_norm']
        stu_ca_v = stu_fea['AV_Trans']['av_v']
        tea_ca_v = tea_fea['AV_Trans']['av_v']

        stu_ca_vv = torch.bmm(stu_ca_v_norm, stu_ca_v.transpose(1, 2))
        tea_ca_vv = torch.bmm(tea_ca_v_norm, tea_ca_v.transpose(1, 2))

        # print(f"stu_ca_vv: {stu_ca_vv}, tea_ca_vv: {tea_ca_vv}")

        sm_stu_ca_vv = self._softmax_w_temperature(stu_ca_vv, temperature)
        sm_tea_ca_vv = self._softmax_w_temperature(tea_ca_vv, temperature)

        if w_softmax:
            return sm_tea_ca_qk, sm_stu_ca_qk, sm_tea_ca_vv, sm_stu_ca_vv
        else:
            return tea_ca_qk, stu_ca_qk, tea_ca_vv, stu_ca_vv

    def _preprocess_cro_att_va_qkv(self, fea_dict, temperature, w_softmax=False):
        stu_fea, tea_fea = fea_dict['stu_fea'], fea_dict['tea_fea']

        stu_ca_qk = stu_fea['AV_Trans']['va_qk'] 
        tea_ca_qk = tea_fea['AV_Trans']['va_qk']

        stu_ca_qk = stu_ca_qk.transpose(1, 2)
        tea_ca_qk = tea_ca_qk.transpose(1, 2)

        sm_stu_ca_qk = self._log_softmax_w_temperature(stu_ca_qk, temperature)
        sm_tea_ca_qk = self._log_softmax_w_temperature(tea_ca_qk, temperature)

        stu_ca_v_norm = stu_fea['AV_Trans']['va_v_norm']
        tea_ca_v_norm = tea_fea['AV_Trans']['va_v_norm']
        stu_ca_v = stu_fea['AV_Trans']['va_v']
        tea_ca_v = tea_fea['AV_Trans']['va_v']

        # print(f"stu_ca_v_norm: {stu_ca_v_norm}, stu_ca_v: {stu_ca_v}")
        # print(f"tea_ca_v_norm: {tea_ca_v_norm}, tea_ca_v: {tea_ca_v}")

        stu_ca_vv = torch.bmm(stu_ca_v_norm, stu_ca_v.transpose(1, 2))
        tea_ca_vv = torch.bmm(tea_ca_v_norm, tea_ca_v.transpose(1, 2))

        sm_stu_ca_vv = self._softmax_w_temperature(stu_ca_vv, temperature)
        sm_tea_ca_vv = self._softmax_w_temperature(tea_ca_vv, temperature)

        if w_softmax:
            return sm_tea_ca_qk, sm_stu_ca_qk, sm_tea_ca_vv, sm_stu_ca_vv
        else:
            return tea_ca_qk, stu_ca_qk, tea_ca_vv, stu_ca_vv

    
    def _preprocess_qkv(self, fea_dict, temperature, w_softmax=False):
        stu_fea, tea_fea = fea_dict['stu_fea'], fea_dict['tea_fea']

        stu_fus_qk = stu_fea['Fus_Trans']['mem_qk'] 
        tea_fus_qk = tea_fea['Fus_Trans']['mem_qk']

        # print(f"tea_fus_qk: {tea_fus_qk.shape}, stu_fus_qk: {stu_fus_qk.shape}")

        sm_stu_fus_qk = self._softmax_w_temperature(stu_fus_qk, temperature)
        sm_tea_fus_qk = self._softmax_w_temperature(tea_fus_qk, temperature)

        stu_fus_v_norm = stu_fea['Fus_Trans']['mem_v_norm']
        tea_fus_v_norm = tea_fea['Fus_Trans']['mem_v_norm']
        stu_fus_v = stu_fea['Fus_Trans']['men_v']
        tea_fus_v = tea_fea['Fus_Trans']['men_v']

        stu_fus_vv = torch.bmm(stu_fus_v_norm, stu_fus_v.transpose(1, 2))
        tea_fus_vv = torch.bmm(tea_fus_v_norm, tea_fus_v.transpose(1, 2))

        sm_stu_fus_vv = self._softmax_w_temperature(stu_fus_vv, temperature)
        sm_tea_fus_vv = self._softmax_w_temperature(tea_fus_vv, temperature)

        if w_softmax:
            return sm_tea_fus_qk, sm_stu_fus_qk, sm_tea_fus_vv, sm_stu_fus_vv
        else:
            return tea_fus_qk, stu_fus_qk, tea_fus_vv, stu_fus_vv
    
    def _single_qkv(self, fea_dict, states='tea_fea'):
        features = fea_dict[states]
        
        q_norm = features['Fus_Trans']['mem_q_norm']
        k = features['Fus_Trans']['mem_k']
        v_norm = features['Fus_Trans']['mem_v_norm']
        v = features['Fus_Trans']['men_v']
        
        return q_norm, k, v_norm, v
    
    def _repeat_data(self, aB, pB, a_q_norm, a_k, a_v_norm, a_v):

        if aB < pB:
            repeat_dim = pB // aB + 1

            a_q_norm = a_q_norm.repeat(repeat_dim, 1, 1)[:pB, ...]
            a_k = a_k.repeat(repeat_dim, 1, 1)[:pB, ...]
            a_v_norm = a_v_norm.repeat(repeat_dim, 1, 1)[:pB, ...]
            a_v = a_v.repeat(repeat_dim, 1, 1)[:pB, ...]

        return a_q_norm, a_k, a_v_norm, a_v

    def _cross_speaker_att_vv(self, anchor, target, TEMP, states='tea_fea'):
        a_q_norm, a_k, a_v_norm, a_v = self._single_qkv(anchor, states)
        t_q_norm, t_k, t_v_norm, t_v = self._single_qkv(target, states)

        aB, _, _ = a_q_norm.size()
        pB, _, _ = t_q_norm.size()

        a_q_norm, a_k, a_v_norm, a_v = self._repeat_data(aB, pB, a_q_norm, a_k, a_v_norm, a_v)

        att = torch.bmm(a_q_norm, t_k.transpose(1, 2))
        att = self._softmax_w_temperature(att, TEMP)

        vv = torch.bmm(a_v_norm, t_v.transpose(1, 2))
        vv = self._softmax_w_temperature(vv, TEMP)

        return att, vv

    def cross_speaker_loss(self, anchor, target):
        tea_crs_att, tea_crs_vv = self._cross_speaker_att_vv(anchor, target, self.pT, states='tea_fea')
        stu_crs_att, stu_crs_vv = self._cross_speaker_att_vv(anchor, target, self.pT, states='stu_fea')

        # print(f"tea_crs_att: {tea_crs_att.shape}")

        loss = self.pT**2 *  (self.kl_div(stu_crs_att.log(), tea_crs_att) + \
                            self.kl_div(stu_crs_vv.log(), tea_crs_vv))
        
        return loss
    
    def av_consistency_loss(self, anchor):
        a_tea_fus_qk, a_stu_fus_qk, a_tea_fus_vv, a_stu_fus_vv = self._preprocess_qkv(anchor, self.cT, w_softmax=True)

        loss = self.cT**2 *  (self.kl_div(a_stu_fus_qk.log(), a_tea_fus_qk) + \
                            self.kl_div(a_stu_fus_vv.log(), a_tea_fus_vv))

        return loss

    def av_saca_consistency_loss(self, anchor):
        a_tea_fus_qk, a_stu_fus_qk, a_tea_fus_vv, a_stu_fus_vv = self._preprocess_qkv(anchor, self.cT, w_softmax=True)
        a_tea_av_qk, a_stu_av_qk, a_tea_av_vv, a_stu_av_vv = self._preprocess_cro_att_av_qkv(anchor, self.cT, w_softmax=True)
        a_tea_va_qk, a_stu_va_qk, a_tea_va_vv, a_stu_va_vv = self._preprocess_cro_att_va_qkv(anchor, self.cT, w_softmax=True)

        sa_loss = self.cT**2 * (self.kl_div(a_stu_fus_qk.log(), a_tea_fus_qk) + \
                            self.kl_div(a_stu_fus_vv.log(), a_tea_fus_vv))

        av_loss = self.cT**2 * (self.kl_div(a_stu_av_qk.log(), a_tea_av_qk) + \
                            self.kl_div(a_stu_av_vv.log(), a_tea_av_vv))

        # va_loss = self.cT**2 *  (self.kl_div(a_stu_va_qk.log(), a_tea_va_qk) + \
        #                     self.kl_div(a_stu_va_vv.log(), a_tea_va_vv))

        # print(f"before log a_stu_va_qk: {a_stu_va_qk}")
        # print(f"after log a_stu_va_qk: {(a_stu_va_qk + 1**(-16)).log()}")

        va_loss = self.cT**2 * (self.kl_div(a_stu_va_qk, a_tea_va_qk) + \
                                self.kl_div(a_stu_va_vv.log(), a_tea_va_vv))
        # va_loss = self.cT**2 * (self.kl_div(a_stu_va_vv.log(), a_tea_va_vv))
                    # + self.kl_div(a_stu_va_vv, a_tea_va_vv))


        # return sa_loss + av_loss + va_loss
        return sa_loss + av_loss + va_loss
        # return va_loss
        # return sa_loss
        # return sa_loss + av_loss

    def forward(self, anchor, target, states='Normal'):
        

        if states == "XBM":
            cross_loss = self.cross_speaker_loss(anchor, target)
            return cross_loss
        else:
            # av_con_loss = self.av_consistency_loss(anchor)
            av_con_loss = self.av_saca_consistency_loss(anchor)
            return av_con_loss
        # pair_loss = 0
        # av_con_loss = self.av_consistency_loss(anchor)
        # return av_con_loss, pair_loss
        # return 0, 0


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

    sad_kl_Loss = Loss(cT=5, pT=5)
    res_av_con_loss = sad_kl_Loss(input_block, target_block)
    print(f"res_av_con_loss: {res_av_con_loss}")

    res_av_con_loss = sad_kl_Loss(input_block, target_block, states="XBM")
    print(f"res_av_con_loss: {res_av_con_loss}")
    # print(f"res_pair_loss: {res_pair_loss}")
