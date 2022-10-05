from regex import P
import torch
import torch.nn as nn
import torch.nn.functional as F
# from geomloss import SamplesLoss

class Loss(nn.Module):
    """ Distilling the Knowledge using MiniLM."""
    def __init__(self, state):
        super(Loss, self).__init__()
        self.state = state
        print(f"State: {self.state}")
        self.kl_div = nn.KLDivLoss(reduction="mean")
        self.linear = nn.Linear(int(64), int(25)).cuda()
        self.mse_loss = nn.MSELoss()

    def _softmax_w_temperature(self, m, temperature=1):
        return F.softmax(m.float() / temperature, dim=-1).type_as(m)

    def _preprocess_qk(self, fea_dict, temperature, w_softmax=False):
        stu_fea, tea_fea = fea_dict['stu_fea'], fea_dict['tea_fea']

        stu_fus_qk = stu_fea['Fus_Trans']['mem_qk'] 
        tea_fus_qk = tea_fea['Fus_Trans']['mem_qk']

        # print(f"tea_fus_qk: {tea_fus_qk.shape}, stu_fus_qk: {stu_fus_qk.shape}")

        sm_stu_fus_qk = self._softmax_w_temperature(stu_fus_qk, temperature)
        sm_tea_fus_qk = self._softmax_w_temperature(tea_fus_qk, temperature)

        if w_softmax:
            return sm_tea_fus_qk, sm_stu_fus_qk
        else:
            return tea_fus_qk, stu_fus_qk

    def _preprocess_value(self, fea_dict):
        stu_fea, tea_fea = fea_dict['stu_fea'], fea_dict['tea_fea']

        stu_fus_v = stu_fea['Fus_Trans']['men_v']
        tea_fus_v = tea_fea['Fus_Trans']['men_v']

        tea_fus_v = self.linear(tea_fus_v)
        
        return tea_fus_v, stu_fus_v

    def forward(self, anchor):

        a_tea_fus_qk, a_stu_fus_qk = self._preprocess_qk(anchor, temperature=1, w_softmax=True)
        tea_fus_v, stu_fus_v = self._preprocess_value(anchor)


        # if self.state is 'trans':
        loss =  self.kl_div(a_stu_fus_qk.log(), a_tea_fus_qk) + \
                self.mse_loss(stu_fus_v, tea_fus_v)
        
        return loss

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

    sad_kl_Loss = Loss(state="trans")
    res_av_con_loss = sad_kl_Loss(input_block)
    print(f"res_av_con_loss: {res_av_con_loss}")

    res_av_con_loss = sad_kl_Loss(input_block)
    print(f"res_av_con_loss: {res_av_con_loss}")