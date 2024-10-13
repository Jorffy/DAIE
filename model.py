import torch
import math
import os
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from transformers import BertModel, CLIPModel

import gat as tg_conv
from glcl import multihead_contrastive_loss, dep_adj_m

class Text_mlp(nn.Module):
    def __init__(self, input_size=512, out_size=300):
        super(Text_mlp, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.norm = nn.LayerNorm(self.out_size)
        self.linear = nn.Linear(self.out_size, 1)
        self.linear_ = nn.Linear(self.input_size, self.out_size)

    def forward(self, t_lhs, word_seq=None, key_padding_mask=None, lam=1):
        tt = t_lhs[:,1:-1,:]
        captions = []
        for i in range(tt.size(0)):
            # [X,L,H] X is the number of np and word
            captions.append(torch.stack([torch.mean(tt[i][tup[0]:tup[1], :], dim=0) for tup in word_seq[i]]))
        ttt = pad_sequence(captions, batch_first=True).cuda()
        text_output = self.norm(self.linear_(ttt))

        text_score = self.linear(text_output).squeeze(-1).masked_fill_(key_padding_mask, float("-Inf"))
        text_score = nn.Softmax(dim=1)(text_score*lam).unsqueeze(2).repeat((1,1,self.out_size))

        return text_output, text_score

class Image_mlp(nn.Module):
    def __init__(self, input_dim=768, inter_dim=500, output_dim=300):
        super(Image_mlp, self).__init__()
        self.input_dim = input_dim
        self.inter_dim = inter_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.inter_dim)
        self.fc2 = nn.Linear(self.inter_dim, self.output_dim)
        self.fc3 = nn.Linear(self.output_dim, 1)
        self.relu1 = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, v_lhs, lam=1):
        vision_output = self.relu1(self.fc1(v_lhs))
        vision_output = self.relu1(self.fc2(vision_output))
        vision_output = self.norm(vision_output)
        
        pv = self.fc3(vision_output).squeeze()
        pv = self.softmax(pv*lam)

        return vision_output, pv

class CroModelAtt(nn.Module):
    def __init__(self, input_size=300, nhead=6, dim_feedforward=600, dropout=0.2, cro_layer=1):
        super(CroModelAtt, self).__init__()
        self.input_size = input_size
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.cro_layer = cro_layer

        self.co_att = nn.MultiheadAttention(input_size, nhead, dropout=dropout)
        self.linear1 = nn.Linear(input_size, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_size)
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.norm3 = nn.LayerNorm(input_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

    def cro_tranformer(self, q, k, v):
        # LayerNorm
        qq = self.norm1(q)
        kk = self.norm1(k)
        vv = self.norm1(v)
        # MultiheadAttention
        tgt = self.co_att(qq, kk, vv)[0]
        # Add + LayerNorm
        tgt2 = qq + self.dropout1(tgt)
        tgt2 = self.norm2(tgt2)
        # Feed Forward
        inter = self.dropout2(self.relu(self.linear1(tgt2)))
        tgt3 = self.dropout2(self.linear2(inter))
        # Add
        tgt4 = tgt2 + tgt3
        tgt4 = self.norm3(tgt4)
        return tgt4

    def forward(self, t, v, togeter=0):
        t1 = t.permute(1, 0, 2)
        v1 = v.permute(1, 0, 2)
        if togeter==1:
            for i in range(self.cro_layer):
                t2 = self.cro_tranformer(t1,v1,v1)
                v2 = self.cro_tranformer(v1,t1,t1)
                t1 = t2
                v1 = v2
        else:
            for i in range(self.cro_layer):
                t1 = self.cro_tranformer(t1,v1,v1)
        t3 = t1.permute(1, 0, 2)
        v3 = v1.permute(1, 0, 2)
        return t3, v3

class Fusion(nn.Module):
    def __init__(self, input_size=300, txt_gat_layer=2, txt_gat_drop=0.2, txt_gat_head=5, txt_self_loops=False,
                 img_gat_layer=2, img_gat_drop=0.2, img_gat_head=5, img_self_loops=False, is_knowledge=0):
        super(Fusion, self).__init__()
        self.input_size = input_size
        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop = txt_gat_drop
        self.txt_gat_head = txt_gat_head
        self.txt_self_loops = txt_self_loops

        self.img_gat_layer = img_gat_layer
        self.img_gat_drop = img_gat_drop
        self.img_gat_head = img_gat_head
        self.img_self_loops = img_self_loops

        self.txt_conv = nn.ModuleList(
            [tg_conv.GATConv(in_channels=self.input_size, out_channels=self.input_size, heads=self.txt_gat_head,
                             concat=False, dropout=self.txt_gat_drop, fill_value="mean",
                             add_self_loops=self.txt_self_loops, is_text=True)
             for i in range(self.txt_gat_layer)])

        self.img_conv = nn.ModuleList([tg_conv.GATConv(in_channels=self.input_size, out_channels=self.input_size,
                                                           heads=self.img_gat_head, concat=False,
                                                           dropout=self.img_gat_drop, fill_value="mean",
                                                           add_self_loops=self.img_self_loops) for i in
                                           range(self.img_gat_layer)])

        # for token compute the importance of each token
        self.linear1 = nn.Linear(self.input_size, 1)
        # for np compute the importance of each np
        self.linear2 = nn.Linear(self.input_size, 1)
        self.norm = nn.LayerNorm(self.input_size)
        self.relu1 = nn.ReLU()

    def forward(self, t2, v2, edge_index, gnn_mask, score, key_padding_mask, np_mask, img_edge_index, pv,
                gnn_mask_know=None, lam=1):
        # incongruity score of token level
        q1 = torch.bmm(t2, v2.permute(0, 2, 1)) / math.sqrt(t2.size(2))
        pa_token = self.linear1(t2).squeeze(-1).masked_fill_(key_padding_mask, float("-Inf"))
        # (N,token_length)

        # GAT of textual graph
        tnp = t2
        # for node with out edge, it representation will be zero-vector
        gat_txt = [] # gat L_T textual graph after GAT
        for gat in self.txt_conv:
            tnp = self.norm(torch.stack(
                [(self.relu1(gat(data[0], data[1].cuda(), mask=data[2]))) for data in zip(tnp, edge_index, gnn_mask)]))
            gat_txt.append(tnp)

        # GAT of visual graph
        v3 = v2
        gat_img = [] # gat L_I visual graph after GAT
        for gat in self.img_conv:
            v3 = self.norm(torch.stack([self.relu1(gat(data, img_edge_index.cuda())) for data in v3]))
            gat_img.append(v3)

        c = torch.sum(score * t2, dim=1, keepdim=True)
        tnp = torch.cat([tnp, c], dim=1)

        #  incongruity score of graph level
        q2 = torch.bmm(tnp, v3.permute(0, 2, 1)) / math.sqrt(tnp.size(2))
        pa_np = self.linear2(tnp).squeeze(-1).masked_fill_(np_mask, float("-Inf"))

        pa_token = nn.Softmax(dim=1)(pa_token * lam).unsqueeze(2).repeat((1, 1, v3.size(1)))
        pa_np = nn.Softmax(dim=1)(pa_np * lam).unsqueeze(2).repeat((1, 1, v3.size(1)))
        a_1 = torch.sum(q1 * pa_token, dim=1)
        a_2 = torch.sum(q2 * pa_np, dim=1)
        a = torch.cat([a_1, a_2], dim=1)
        pv = pv.repeat(1, 2)
        apv = a * pv

        return apv, gat_txt, gat_img

class DAIE_Model(nn.Module):
    def __init__(self, args, clip_path):
        super(DAIE_Model, self).__init__()
        self.args = args
        self.clip_model = CLIPModel.from_pretrained(clip_path)
        self.clip_img_pri_model = CLIPModel.from_pretrained(clip_path)

        self.text_mlp = Text_mlp(input_size=512, out_size=args.mlp_output)
        self.img_mlp = Image_mlp(input_dim=768, inter_dim=500, output_dim=args.mlp_output)
        self.img_mlp_pri = Image_mlp(input_dim=768, inter_dim=500, output_dim=args.mlp_output)
        self.cro_model_att = CroModelAtt(input_size=args.mlp_output, nhead=6, dim_feedforward=2*args.mlp_output, dropout=self.args.dropout, cro_layer=1)
        self.fusion = Fusion(txt_gat_layer=args.gat_layer, txt_gat_drop=self.args.dropout, img_gat_layer=args.gat_layer,img_gat_drop=self.args.dropout)

        self.linear_att = nn.Linear(in_features=50, out_features=1)
        self.tanh = nn.Tanh()
        self.linear_att_pri = nn.Linear(in_features=50, out_features=1)
        self.tanh_pri = nn.Tanh()

        self.linear_classifier1 = nn.Linear(in_features=2*args.mlp_output, out_features=2)
        self.linear_classifier2 = nn.Linear(in_features=50 * 2, out_features=2)

        self.logit_scale1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.01))
        self.logit_scale_pri1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.01))

        self.loss_fn = nn.CrossEntropyLoss()

        self.lam = args.lam
        self.beta = args.beta # the weight of loss_tlcl
        self.theta = args.theta # the weight of loss_glcl
        self.gamml = args.gamml # the weight of loss_imgs_pri
        self.imgpri = args.imgpri # have TLCL_PRI Yes(True) or No(False)
        self.nsw = args.nsw # have the GLCL_NSW Yes(True) or No(False)
        self.temperature = args.temperature

    def forward(self, texts, imgs, imgs_pri, labels, \
            word_spans, word_len, word_mask, dep_se, gnn_mask, np_mask, img_mask, img_edge_index):

        bsz = texts["input_ids"].size(0)
        # CLIP
        clip_model = self.clip_model(input_ids=texts["input_ids"],attention_mask=texts["attention_mask"],pixel_values=imgs,return_loss=True)
        clip_text_model_output, clip_vision_model_output = clip_model.text_model_output, clip_model.vision_model_output

        # Feature Encoder
        # MLP
        clip_text_output_hidden = clip_text_model_output.last_hidden_state
        clip_vision_output_hidden = clip_vision_model_output.last_hidden_state
        texts_mlp, texts_score = self.text_mlp(clip_text_output_hidden, word_seq=word_spans,
                                        key_padding_mask=word_mask, lam=self.lam)
        imgs_mlp, pv = self.img_mlp(clip_vision_output_hidden, lam=self.lam)
        # Cross-modal Transformer
        texts_att, imgs_att = self.cro_model_att(texts_mlp, imgs_mlp, togeter=0) # T V
        # lower dimensionality
        self.linear_texts_att = nn.Linear(in_features=texts_att.size(1), out_features=1).cuda()
        texts_att_lin = self.tanh(self.linear_texts_att(texts_att.permute(0, 2, 1)).squeeze())
        imgs_att_lin = self.tanh(self.linear_att(imgs_att.permute(0, 2, 1)).squeeze())
        texts_att_lin = texts_att_lin / texts_att_lin.norm(dim=1, keepdim=True)
        imgs_att_lin = imgs_att_lin / imgs_att_lin.norm(dim=1, keepdim=True)

        # apv = P_TLF + P_GLF, gat_texts is the L_T-th textual graph after GAT
        apv, gat_texts, gat_imgs = self.fusion(t2=texts_att, v2=imgs_att, edge_index=dep_se, gnn_mask=gnn_mask, score=texts_score,
                                        key_padding_mask=word_mask, np_mask=np_mask, img_edge_index=img_edge_index, pv=pv,
                                        lam=self.lam)

        # Token-Level Fusion's TLCL
        imgs_PRI = self.get_patch_based_reconstructed_image(bsz=bsz, texts=texts, imgs_pri=imgs_pri)
        loss_tlcl = self.tlcl(bsz, imgs_att_lin, texts_att_lin, labels, imgs_PRI, temperature=self.temperature)

        # Graph-Level Fusion's GLCL
        # textual graph
        texts_len = texts_mlp.size(1)
        loss_glcl_t = self.glcl(bsz=bsz, gat_txt=gat_texts, texts_len=texts_len, dep_se=dep_se, word_len=word_len, tau=1, input_type='texts')
        # visual graph
        imgs_len = imgs_mlp.size(1)
        loss_glcl_v = self.glcl(bsz=bsz, gat_txt=gat_imgs, texts_len=imgs_len, dep_se=img_edge_index, word_len=None, tau=1, input_type='imgs')

        # Prediction
        y = 0.5*self.linear_classifier1(torch.cat([texts_att_lin, imgs_att_lin], dim=1)) + 0.5*self.linear_classifier2(torch.cat([apv], dim=1))

        loss = self.loss_fn(y, labels.view(-1)) + self.beta*loss_tlcl + self.theta*0.5*(loss_glcl_t+loss_glcl_v)

        return loss, y

    def tlcl(self, bsz, image_features, text_features, labels, imgs_PRI, temperature=1):
        # TLCL
        # cosine similarity as logits
        logit_scale = self.logit_scale1.exp()  # learnable parameter

        # multiply the features to obtain similarity
        q_v_image = logit_scale * image_features @ text_features.t()
        q_t_text = q_v_image.t()  # transpose
        # shape = [batch_size, batch_size]

        # labels_cl = torch.zeros(bsz, dtype=torch.long).cuda()
        labels_cl = torch.arange(bsz, device=q_v_image.device)

        if self.imgpri:
            logit_scale_rpi = self.logit_scale_pri1.exp()  # learnable parameter (RPI)
            loss_imgs_rpi = 0.
            for imgs_pri in imgs_PRI:
                q_v_image_pri = logit_scale_rpi * imgs_pri @ text_features.t()
                q_t_text_pri = q_v_image_pri.t()
                # labels_cl_pri = torch.zeros(bsz, dtype=torch.long).cuda()
                labels_cl_pri = torch.arange(bsz, device=q_v_image_pri.device)

                loss_imgs_rpi += (F.cross_entropy(q_v_image_pri / temperature, labels_cl_pri) +
                                    F.cross_entropy(q_t_text_pri / temperature, labels_cl_pri)) / 2
            loss_imgs_rpi = loss_imgs_rpi/imgs_PRI.size(0)

        total_loss = (F.cross_entropy(q_v_image / temperature, labels_cl) +
                        F.cross_entropy(q_t_text / temperature, labels_cl)) / 2

        if self.imgpri:
            total_loss = total_loss + self.gamml*loss_imgs_rpi

        return total_loss

    def get_patch_based_reconstructed_image(self, bsz, texts, imgs_pri):
        # patch-based reconstructed image (RPI)
        imgs_pri_len = len(imgs_pri)
        imgs_pri = imgs_pri.permute(1, 0, 2, 3, 4)
        imgs_reconstructed = torch.empty(1,imgs_pri_len,self.args.mlp_output).cuda()
        for i in range(imgs_pri.size(0)):
            imgs_pri_1 = self.clip_img_pri_model(input_ids=texts["input_ids"],attention_mask=texts["attention_mask"],pixel_values=imgs_pri[i]).vision_model_output.last_hidden_state
            imgs_pri_1, _ = self.img_mlp_pri(imgs_pri_1, lam=self.lam)
            imgs_pri_1 = self.tanh_pri(self.linear_att_pri(imgs_pri_1.permute(0, 2, 1)).squeeze())
            imgs_pri_1 = imgs_pri_1 / imgs_pri_1.norm(dim=1, keepdim=True)
            imgs_pri_1 = imgs_pri_1.unsqueeze(0)

            imgs_reconstructed = torch.cat([imgs_reconstructed, imgs_pri_1], dim=0)
        imgs_reconstructed = imgs_reconstructed[1:,:,:]

        return imgs_reconstructed

    def glcl(self, bsz, gat_txt, texts_len, dep_se, word_len, tau=1, input_type='texts'):
        # GLCL
        # Adjacency matrix
        c_adj = dep_adj_m(edge_index=dep_se, max_len=texts_len, input_type=input_type).cuda()

        gat_bs = []
        for i in range(bsz):
            gat_bbs = []
            for gat in gat_txt:
                gat_bbs.append(gat[i])
            gat_bbs = torch.stack(gat_bbs)
            gat_bs.append(gat_bbs)
        gat_loss = 0.
        for i in range(bsz):
            if input_type == 'texts':
                gat_loss = gat_loss + multihead_contrastive_loss(gat_bs[i], adj=c_adj[i], word_len=word_len[i], tau=tau, nsw=self.nsw)
            else:
                gat_loss = gat_loss + multihead_contrastive_loss(gat_bs[i], adj=c_adj[0], word_len=None, tau=tau, nsw=self.nsw)
        gat_loss = gat_loss/bsz
        # gat_loss = 0
        return gat_loss




