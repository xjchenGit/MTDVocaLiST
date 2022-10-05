import torch
from torch import device, nn
from torch.nn import functional as F
from models.transformer_encoder import TransformerEncoder
from models.conv import Conv2d, Conv3d


class SyncTransformer(nn.Module):
    def __init__(self, d_model=512):
        super(SyncTransformer, self).__init__()
        self.d_model = d_model
        # layers = [32, 64, 128, 256, 512]
        layers = [16, 32, 64, 128, d_model]
        # layers = [16, 32, 64, 128, ]
        self.vid_prenet = nn.Sequential(
            Conv3d(3, layers[0], kernel_size=7, stride=1, padding=3),

            Conv3d(layers[0], layers[1], kernel_size=5, stride=(1, 2, 1), padding=(1, 1, 2)),
            Conv3d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),

            Conv3d(layers[1], layers[2], kernel_size=3, stride=(2, 2, 1), padding=1),
            Conv3d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),

            Conv3d(layers[2], layers[3], kernel_size=3, stride=(2, 2, 1), padding=1),
            Conv3d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),

            Conv3d(layers[3], layers[4], kernel_size=3, stride=(2, 2, 1), padding=1),
            Conv3d(layers[4], layers[4], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[4], layers[4], kernel_size=3, stride=1, padding=1, residual=True),

            Conv3d(layers[4], layers[4], kernel_size=3, stride=(2, 2, 1), padding=1),
            Conv3d(layers[4], layers[4], kernel_size=3, stride=1, padding=(0, 0, 1)),
            Conv3d(layers[4], layers[4], kernel_size=1, stride=1, padding=0),)

        self.aud_prenet = nn.Sequential(
            Conv2d(1, layers[0], kernel_size=3, stride=1, padding=1),
            Conv2d(layers[0], layers[0], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[0], layers[0], kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(layers[0], layers[1], kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(layers[1], layers[2], kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(layers[2], layers[3], kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(layers[3], layers[4], kernel_size=3, stride=1, padding=(0, 1)),
            Conv2d(layers[4], layers[4], kernel_size=1, stride=1, padding=0), )

        self.av_transformer = TransformerEncoder(embed_dim=d_model,
                                                 num_heads=8,
                                                 layers=4,
                                                 attn_dropout=0.0,
                                                 relu_dropout=0.1,
                                                 res_dropout=0.1,
                                                 embed_dropout=0.25,
                                                 attn_mask=True)

        self.va_transformer = TransformerEncoder(embed_dim=d_model,
                                                 num_heads=8,
                                                 layers=4,
                                                 attn_dropout=0.0,
                                                 relu_dropout=0.1,
                                                 res_dropout=0.1,
                                                 embed_dropout=0.25,
                                                 attn_mask=True)

        self.mem_transformer = TransformerEncoder(embed_dim=d_model,
                                                  num_heads=8,
                                                  layers=4,
                                                  attn_dropout=0.0,
                                                  relu_dropout=0.1,
                                                  res_dropout=0.1,
                                                  embed_dropout=0.25,
                                                  attn_mask=True)

        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(d_model, 1)
        self.feature_list = {}

    def forward(self, frame_seq, mel_seq):
        # Shape frame seq: [4, 1, 80, 80], mel_seq: [4, 75, 48, 96]
        
        """ Output:
        aud_embedding: torch.Size([4, 512, 1, 80])
        vid_embedding: torch.Size([4, 512, 1, 1, 25])
        """
        B = frame_seq.shape[0]
        vid_embedding = self.vid_prenet(frame_seq.view(B,-1,3,48,96).permute(0,2,3,4,1).contiguous())
        aud_embedding = self.aud_prenet(mel_seq)
        self.feature_list['CNN_feature'] = {'aud': aud_embedding, 'vis': vid_embedding}
        # print(f"[layer 1] aud_embedding: {aud_embedding.shape}, vid_embedding: {vid_embedding.shape}")

        """  Output:
        aud_embedding: torch.Size([4, 512, 80])
        vid_embedding: torch.Size([4, 512, 25])
        """
        vid_embedding = vid_embedding.squeeze(2).squeeze(2)
        aud_embedding = aud_embedding.squeeze(2)
        # print(f"[layer 2] aud_embedding: {aud_embedding.shape}, vid_embedding: {vid_embedding.shape}")

        """ Output:
        aud_embedding: torch.Size([80, 4, 512])
        vid_embedding: torch.Size([25, 4, 512])
        """
        # print(f"vid_embedding: {vid_embedding.shape}")
        vid_embedding = vid_embedding.permute(2, 0, 1).contiguous()
        aud_embedding = aud_embedding.permute(2, 0, 1).contiguous()
        # print(f"[layer 3] aud_embedding: {aud_embedding.shape}, vid_embedding: {vid_embedding.shape}")
        
        """ Output:
        av_embedding: torch.Size([80, 4, 512])
        va_embedding: torch.Size([25, 4, 512])
        """
        av_embedding, av_qk, av_q_norm, av_k, av_v_norm, av_v  = self.av_transformer(aud_embedding, vid_embedding, vid_embedding)
        va_embedding, va_qk, va_q_norm, va_k, va_v_norm, va_v, = self.va_transformer(vid_embedding, aud_embedding, aud_embedding)
        self.feature_list['AV_Trans'] = {'av_emb': av_embedding, 'va_emb': va_embedding, 'av_qk': av_qk, 'va_qk': va_qk, 
                                        'av_q_norm': av_q_norm, 'va_q_norm': va_q_norm, 'av_k': av_k, 'va_k': va_k,
                                        'av_v_norm': av_v_norm, 'va_v_norm': va_v_norm, 'av_v': av_v, 'va_v': va_v}
        # print(f"[layer 4] av_embedding: {av_embedding.shape}, va_embedding: {va_embedding.shape}")

        # cro_vv = torch.bmm(av_v, va_v.transpose(1, 2))
        # cro_vv = F.softmax(cro_vv.float(), dim=-1).type_as(cro_vv)

        """ Output 
        tranformer_out: torch.Size([80, 4, 512])
        t: 80 """
        tranformer_out, men_qk, mem_q_norm, mem_k, men_v_norm, men_v = self.mem_transformer(av_embedding, va_embedding, va_embedding)
        self.feature_list['Fus_Trans'] = {'mem_emb': tranformer_out, 'mem_qk': men_qk, 'mem_q_norm': mem_q_norm, 'mem_k': mem_k,
                                        'mem_v_norm': men_v_norm, 'men_v': men_v}
        # men_vv = torch.bmm(men_v_norm, men_v.transpose(1, 2))
        # men_vv = F.softmax(men_vv.float(), dim=-1).type_as(men_vv)
        t = av_embedding.shape[0]
        # print(f"[layer 5] tranformer_out: {tranformer_out.shape}, t: {t}")

        """ 
        out: torch.Size([4, 512]) 
        h_pooled: torch.Size([4, 512])
        logits_clsf: torch.Size([4, 1])
        """
        out = F.max_pool1d(tranformer_out.permute(1, 2, 0).contiguous(), t).squeeze(-1)
        h_pooled = self.activ1(self.fc(out))  # [batch_size, d_model]
        logits_clsf = (self.classifier(h_pooled))
        return logits_clsf.squeeze(-1), self.feature_list



# Test Model
if __name__ == "__main__":
    model = SyncTransformer(d_model=200)

    # Total trainable params 7.75 M
    # vid_prenet trainable params 4.85 M
    # aud_prenet trainable params 0.46 M
    # av_transformer trainable params 0.79 M
    # va_transformer trainable params 0.79 M
    # mem_transformer trainable params 0.79 M
    # fc trainable params 0.07 M

    # print(f"vid_prenet: {model.vid_prenet}")
    print('Total trainable params {:.2f} M'.format((sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)))
    print('vid_prenet trainable params {:.2f} M'.format((sum(p.numel() for p in model.vid_prenet.parameters() if p.requires_grad) / 1e6)))
    print('aud_prenet trainable params {:.2f} M'.format((sum(p.numel() for p in model.aud_prenet.parameters() if p.requires_grad) / 1e6)))
    print('av_transformer trainable params {:.2f} M'.format((sum(p.numel() for p in model.av_transformer.parameters() if p.requires_grad) / 1e6)))
    print('va_transformer trainable params {:.2f} M'.format((sum(p.numel() for p in model.va_transformer.parameters() if p.requires_grad) / 1e6)))
    print('mem_transformer trainable params {:.2f} M'.format((sum(p.numel() for p in model.mem_transformer.parameters() if p.requires_grad) / 1e6)))
    print('fc trainable params {:.2f} M'.format((sum(p.numel() for p in model.fc.parameters() if p.requires_grad) / 1e6)))

    mel_seq = torch.rand([20, 1, 80, 80])
    frame_seq = torch.rand([20, 15, 48, 96])
    output, stu_feature_list = model(frame_seq, mel_seq)
    # print(f"output: {output.shape}, stu_feature_list: {stu_feature_list['CNN_feature']['aud'].shape}")

    # print(f"output: {output.shape}")
    print(f"Fus_Trans mem_emb: {stu_feature_list['Fus_Trans']['mem_emb'].shape}")
    print(f"Fus_Trans mem_qk: {stu_feature_list['Fus_Trans']['mem_qk'].shape}")
    print(f"Fus_Trans mem_v_norm: {stu_feature_list['Fus_Trans']['mem_v_norm'].shape}")
    print(f"Fus_Trans men_v: {stu_feature_list['Fus_Trans']['men_v'].shape}")
