import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from models.transformers.token_embedding import MixEmbedding
from models.transformers.tools import *
from einops import rearrange, repeat
from models.transformers.transformer_decoder import TransformerDecoder, TransformerEncoder
from models.transformers.transformer_xl_decoder import TrmXLDecoder
import clip
from torch.distributions.categorical import Categorical

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, emb_dropout_prob=0.1):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        self.poseEmbedding = nn.Sequential(
            nn.Linear(self.input_feats, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, self.latent_dim),
            nn.Dropout(emb_dropout_prob),
        )

    def forward(self, x):
        # [bs, ntokens, input_feats]
        # x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x
    
class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        # self.dense = nn.Linear(latent_dim, latent_dim)
        # self.transform_act_fn = F.gelu
        # self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states = self.dense(hidden_states)
        # hidden_states = self.transform_act_fn(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  #(b, s, c)
        # output = output.permute(1, 2, 0)  # [bs, c, seqlen]
        return output


def PE1d_sincos(seq_length, dim):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                         -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(1)


class PositionEmbedding(nn.Module):
    """
    Absolute pos embedding (standard), learned.
    """
    def __init__(self, seq_length, dim, dropout, grad=False):
        super().__init__()
        self.embed = nn.Parameter(data=PE1d_sincos(seq_length, dim), requires_grad=grad)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, prefix_len=0):
        # x.shape: bs, seq_len, feat_dim
        l = x.shape[1] + prefix_len
        x = x.permute(1, 0, 2) + self.embed[prefix_len:l].expand(x.permute(1, 0, 2).shape)
        x = self.dropout(x.permute(1, 0, 2))
        return x


class TransformotionEmbedding(nn.Module):
    def __init__(self, vocab_size, max_position_len, token_type_size, embed_dim, dropout_prob, device='cpu'):
        super(TransformotionEmbedding, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.segment_embeddings = nn.Embedding(token_type_size, embed_dim)
        # self.position_embeddings = nn.Embedding(max_position_len, embed_dim)
        self.position_embeddings = PositionEmbedding(500, embed_dim, dropout_prob)
        self.LayerNorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        self.device = device
        # self.prompt_ln = nn.Linear(768, embed_dim)
        self.prompt_ln = nn.Linear(512, embed_dim) #CLIP
        # self.prompt_ln = nn.Sequential(
        #     nn.Linear(768, 4 * embed_dim),
        #     nn.GELU(),
        #     nn.Linear(4 * embed_dim, embed_dim),
        #     nn.Dropout(dropout_prob),
        # )
        # self.init_weight()
        
    # def init_weight(self):
    #     # 初始化 token_embeddings 权重
    #     nn.init.xavier_uniform_(self.token_embeddings.weight)
        
    #     # 初始化 segment_embeddings 权重
    #     # nn.init.xavier_uniform_(self.segment_embeddings.weight)
        
    #     # 初始化 position_embeddings 权重
    #     nn.init.xavier_uniform_(self.position_embeddings.weight)

    def forward(self, input_ids, segment_ids, prefix_len=0):
        seq_length = input_ids.size(1)
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) + prefix_len
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids) if prefix_len > 0 else position_ids.unsqueeze(0).expand(input_ids.size(0), input_ids.size(1))
        # print(f"clip feat shape{input_ids.shape}")
        token_embeddings = self.token_embeddings(input_ids) if prefix_len > 0 else self.prompt_ln(input_ids).unsqueeze(1) # 如果prefix len 是0 证明是prompt
        # segment_embeddings = self.segment_embeddings(segment_ids)
        # segment_ids = torch.ones(1, input_ids.size(1) - prefix_len, dtype=torch.long).to(self.device) if prefix_len > 0 else torch.zeros(1, input_ids.size(1), dtype=torch.long).to(self.device)
        # segment_embeddings = self.segment_embeddings(segment_ids.to(self.device))
        position_embeddings = self.position_embeddings(token_embeddings, prefix_len) if prefix_len > 0 else self.position_embeddings(token_embeddings)
        # embeddings = token_embeddings + segment_embeddings + position_embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
        

def init_weight(weight):
    nn.init.normal_(weight, 0.0,0.02)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, 0.01)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, 0.01)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TrmXLDecoder') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


class Transformotion(nn.Module):
    def __init__(self, code_dim, vq_model,  latent_dim=1024, ff_size=1024, num_layers=6, clip_dim=512,clip_version=None,
                 num_heads=16, dropout=0.15, opt=None, prompt_drop_prob=0.3, **kargs) -> None:
        super(Transformotion, self).__init__()
        self.code_dim = code_dim
        self.dropout = dropout
        self.clip_dim = clip_dim
        self.vq_model = vq_model
        self.latent_dim = latent_dim
        # self.cond_mode = cond_mode
        self.cond_drop_prob = 0.1
        self.device = opt.device
        _num_tokens = opt.num_tokens + 2 # for motion pad and end
        print(f"opt.num tokens ====={opt.num_tokens}")
        self.num_tokens = _num_tokens
        self.pad_id = opt.num_tokens + 1
        self.opt = opt
        self.num_heads = num_heads
        self.prompt_drop_prob = prompt_drop_prob
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        # self.output_process = OutputProcess(out_feats=opt.num_tokens, latent_dim=latent_dim)
        # self.tok_emb = nn.Embedding(opt.num_tokens + 2, self.latent_dim)
        # self.cond_emb = nn.Linear(clip_dim, self.latent_dim)
        # self.pos_embed = PositionEmbedding(300, self.latent_dim, 0.0, False)
        # self.head = nn.Linear(self.latent_dim, opt.num_tokens, bias=False)
        # self.mix_emb = MixEmbedding(vq_model, self.device)
        # self.prompt_embedding = TransformotionEmbedding(768, self.latent_dim)
        # self.embedding = TransformotionEmbedding(self.num_tokens,
        #                                          max_position_len=opt.max_motion_length // 4 + 2, 
        #                                          token_type_size=2, 
        #                                          embed_dim=self.code_dim, 
        #                                          dropout_prob=0.1, device=self.device)
        
        # self.seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
        #                                                        nhead=self.num_heads,
        #                                                        dim_feedforward=ff_size,
        #                                                        dropout=dropout,
        #                                                        activation='gelu').to(self.device)
        # self.seqTransEncoder = nn.TransformerEncoder(self.seqTransEncoderLayer,
        #                                              num_layers=num_layers).to(self.device)
        # self.seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
        #                                                        nhead=num_heads,
        #                                                        dim_feedforward=ff_size,
        #                                                        dropout=dropout,
        #                                                        activation='gelu').to(self.device)

        # self.seqTransDecoder = nn.TransformerDecoder(self.seqTransDecoderLayer,
        #                                              num_layers=num_layers).to(self.device)
        self.seqTransEncoder = TransformerEncoder(embed_size=self.latent_dim,
                                                  heads=num_heads,
                                                  dropout=dropout,
                                                     num_layers=num_layers).to(self.device)
        self.seqTransDecoder = TransformerDecoder(embed_size=self.latent_dim,
                                                  heads=num_heads,
                                                  dropout=dropout,
                                                     num_layers=num_layers).to(self.device)
        
        cutoffs, tie_projs = [], [False]
        
        self.decoder_xl_dim = 1024
        self.cond_emb = nn.Linear(clip_dim, self.decoder_xl_dim)
        self.head = nn.Linear(self.decoder_xl_dim, opt.num_tokens, bias=False)
        self.seqTransDecoderXL = TrmXLDecoder(self.num_tokens, num_layers, num_heads,
                            self.decoder_xl_dim, d_head=self.decoder_xl_dim // num_heads, d_inner=1024*4, dropout=dropout,
                            dropatt=dropout, tie_weight=True, 
                            d_embed=1024, div_val=1, 
                            tie_projs=tie_projs, pre_lnorm=False,
                            tgt_len=210, ext_len=210, mem_len=210,
                            cutoffs=cutoffs).to(self.device)
        
        self.apply(self.__init_weights)
        self.seqTransDecoderXL.apply(weights_init)
        self.seqTransDecoderXL.word_emb.apply(weights_init)
        self.seqTransDecoderXL.train()
        
        self.mems = tuple()

        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)
        # Add a linear layer to convert the output to vocab_size
        # self.output_layer = nn.Linear(self.code_dim, self.vocab_size)
        # token_embed_weight = torch.normal(mean=0, std=0.02,
        #                                           size=(opt.num_quantizers - 1, self.vocab_size, code_dim))
        # self.token_embed_weight = nn.Parameter(token_embed_weight)

       
    
    def load_and_freeze_token_emb(self, codebook):
        '''
        :param codebook: (c, d)
        :return:
        '''
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)) #add two dummy tokens, 0 vectors
        self.token_emb.requires_grad_(False)
        # self.token_emb.weight.requires_grad = False
        # self.token_emb_ready = True
        print("Token embedding initialized!")
        
    def mask_motion_token(self, motion_ids):
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(0.5 * torch.ones(motion_ids.shape,
                                                         device=self.device))
        mask = mask.round().to(dtype=torch.int64)
        r_indices = torch.randint_like(motion_ids, self.opt.num_tokens)
        a_indices = mask*motion_ids+(1-mask)*r_indices
        return a_indices
        # return motion_ids
        
        
    def mask_prompt(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond
        # bs,seqlen, dim =  cond.shape
        # if force_mask:
        #     return torch.zeros_like(cond)
        # elif self.training:
        #     proba = np.random.rand(1)[0]
        #     bernoulli_mask = torch.bernoulli(proba * torch.ones(bs, seqlen, 1))

        #     # 将掩码扩展到 (bs, seqlen, dim) 的形状
        #     bernoulli_mask = bernoulli_mask.expand(-1, -1, dim).to(self.device)
        #     masked_seq = cond * bernoulli_mask
        #     return masked_seq
        # # elif self.training and self.cond_drop_prob > 0.:
        # #     mask = torch.bernoulli(torch.ones((bs, seq, 1), device=cond.device) * self.cond_drop_prob)
        # #     return cond * (1. - mask)
        # else:
        #     return cond
    
    
    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cuda',
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    
    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text
        # input_ids, feat_clip_text = self.mix_emb.encode_text(raw_text)
        # return input_ids, feat_clip_text
    
    def trans_forward(self, xseq, motion_ids, m_lens, is_generating=False):
        # tgt_mask = self.generate_special_token_mask(xseq.size(0), xseq.size(1), xseq.size(1)-motion_ids.size(1)).to(self.device)
        # tgt_mask = self.generate_square_subsequent_mask(xseq.size(0), xseq.size(1))
        # print(f"xseq shape=========>{xseq.shape}\n tgt mask shape======+>{tgt_mask.shape}")
        # memory = self.seqTransEncoder(xseq)
        # output =  self.seqTransDecoder(memory, pad_token_mask=pad_token_mask)[:,-1:,:] if is_generating else self.seqTransDecoder(memory, pad_token_mask=pad_token_mask)[:,-motion_ids.size(1):,:]
        output =  self.seqTransDecoder(xseq)
        # print(f"output size==========> {output.shape}, \n motion_ids shape: {motion_ids.shape}")
        # output = self.seqTransDecoder(x, memory=memory, tgt_mask=tgt_mask)[1:] #(seqlen, b, e)
        logits = self.head(output) # (b, ntoken, seqlen)
        # motion_logits = logits[:,:,-labels.size(1):]
        # print(f"output size==========> {output}, \n motion_ids shape: {motion_ids.shape}")
        if is_generating is False:
            ce_loss, pred_id, acc = cal_perfor(logits, motion_ids, m_lens, self.pad_id)
            # print(f"pred_id===================+: {pred_id}\n motion_ids==========>{motion_ids}")
            return ce_loss, acc, pred_id, output, logits
        else:
            return logits

    
    def forward(self, prompt_texts, motion_ids, m_lens, labels=None, mems=tuple(), is_generating=False):
        # # input_ids, prompt_logits = self.encode_text(prompt_texts)x
        prompt_logits = self.encode_text(prompt_texts)
        prompt_logits = self.cond_emb(prompt_logits)
        if labels is None:
            labels = motion_ids
        # # pad_token_mask = (input_ids != self.mix_emb.pad_token_id).float()
        # # prompt = self.mask_prompt(prompt_logits, force_mask=False)
        if is_generating == False and self.training:
            bs, ntokens = motion_ids.shape
            # prompt_logits = self.mask_prompt(prompt_logits, force_mask=False)
            # non_pad_mask = lengths_to_mask(m_lens, ntokens)
            # motion_ids = torch.where(non_pad_mask, motion_ids, self.pad_id)
            labels = motion_ids
            motion_ids = motion_ids[:, :-1]
            # print(f"pad id====================================={self.pad_id}\n labelsssss====================================={labels}")
            motion_ids = self.mask_motion_token(motion_ids)
        # print(f"prompt_logits=====>{prompt_logits.shape}")
        # mems = mems
        mems=tuple()
        ret, pred_hid = self.seqTransDecoderXL(is_generating, prompt_logits.unsqueeze(0), motion_ids.permute(1, 0), labels.permute(1, 0), *mems)
        loss, mems = ret[0], ret[1:]
        output = pred_hid
        output = output.permute(1, 0, 2)
        logits = self.head(output)
        if is_generating is False:
            # self.mems = mems
            ce_loss, pred_id, acc = cal_perfor(logits, labels, m_lens, self.pad_id)
            # loss = loss.float().mean().type_as(loss)
            return ce_loss, acc, pred_id, output, logits, mems
        else:
            return logits, mems
        # if is_generating == True:
        #     logits = self.trans_forward(xseq, motion_ids, m_lens, is_generating=is_generating)
        #     return logits
        # else:    
        #     ce_loss, acc, pred_id, output, logits = self.trans_forward(xseq, motion_ids, m_lens, is_generating=is_generating)
    
        #     return ce_loss, acc, pred_id, output, logits



    
    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    
    def generate_square_subsequent_mask(self, bs, sz):
        mask = (torch.triu(torch.ones(sz, sz, device=self.device)) == 1).transpose(0, 1)
        mask = mask.float()
        # Expand mask to fit multi-head attention format [1, size, size]
        mask = mask.unsqueeze(0).unsqueeze(1).repeat(bs, 1, 1, 1)
        
        # print(f"training mask=======++>{mask}")
        return mask

    def generate_special_token_mask(self, bs, size, pos):
        """
        生成一个结合特殊标记限制的自回归掩码。
        模型可以看到特殊标记位置之前的所有token，但从特殊标记开始，只能看到当前位置及之前的位置。

        参数:
        size (int): 掩码的大小，即序列的长度。
        som_token_position (list): 包含每个样本中 `[SOM]` token 位置索引的列表。

        返回:
        torch.Tensor: 结合特殊标记限制的自回归掩码矩阵，形状为 [bs * nheads, size, size]。
        """
        # bs = len(som_token_pos)
        mask = torch.ones(size, size, dtype=torch.float, device=self.device)
        lower_triangular_mask = torch.tril(torch.ones(size - pos - 1, size - pos - 1, device=self.device), diagonal=0)
        lower_triangular_mask = lower_triangular_mask.float()
        mask[0:pos, 0:pos] = torch.ones((pos, pos), device=self.device)
        mask[pos + 1:, pos + 1:] = lower_triangular_mask

        # for i in range(bs):
        #     pos = int(som_token_pos[i])
        #     assert pos < size, "special_token_position must be within the sequence length."

        #     lower_triangular_mask = torch.tril(torch.ones(size - pos - 1, size - pos - 1, device=self.device), diagonal=0)
        #     lower_triangular_mask = lower_triangular_mask.float().masked_fill(lower_triangular_mask == 0, float('-inf')).masked_fill(lower_triangular_mask == 1, float(0.0))
        #     mask[i, pos + 1:, pos + 1:] = lower_triangular_mask

        # Expand mask to fit multi-head attention format [bs, nheads, size, size]
        # mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(bs * self.num_heads, size, size)
        mask = mask.unsqueeze(0).unsqueeze(1).repeat(bs, 1, 1, 1)
        # print(f"mask====={mask}")
        # print(f"gen mask=======++>{mask}")
        return mask
    
    
    # def forward_with_cond_scale(self,
    #                             prompts,
    #                             motion_ids,
                                
    #                             cond_scale=3,
    #                             force_mask=False):
    #     # bs = motion_ids.shape[0]
    #     # if cond_scale == 1:
    #     if force_mask:
    #         return self.foward()

    #     logits = self.trans_forward(motion_ids, cond_vector, padding_mask)
    #     if cond_scale == 1:
    #         return logits

    #     aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

    #     scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
    #     return scaled_logits
    
    @torch.no_grad()
    @eval_decorator
    def generate(self, prompt_texts, m_lens, labels, temperature=0.6, tk=1, topk_filter_thres=0.9, cond_scale=3):
        self.eval()
        self.seqTransDecoderXL.eval()
        seq_len = max(m_lens).to(self.device)
        batch_size = len(m_lens)
    
        # input_ids, prompt_logits = self.encode_text(prompt_texts)
        # prompt_logits = self.encode_text(prompt_texts)
        # # print(f"prompt_logits====>{prompt_logits.shape}")
        # # pad_token_mask = (input_ids != self.mix_emb.pad_token_id).float()
        # # print(f"pad token mask{pad_token_mask}")
        # prompt = prompt_logits
        # # prompt1 = self.mask_prompt(prompt_logits, force_mask=True)
        # # x = self.embedding(motion_ids)
        # prompt = self.embedding(prompt, torch.zeros((batch_size, 1), dtype=torch.long)).to(self.device)
        # prompt = self.input_process(prompt)
        # # print(prompt)
        # # x = self.input_process(x)
        res_seq_ids = []
        # # _som_token_ids = torch.full((batch_size, 1), self.pad_id, device=self.device)
        # # som_token_ids = self.embedding(_som_token_ids)
        # # som_token_ids = self.input_process(som_token_ids)
        # generated = torch.cat([prompt], dim=1).clone()
        mems = tuple()
        
        # segment_ids = torch.ones_like(generated)
        generated = torch.empty(batch_size, 0, dtype=torch.long).to(self.device)
        for k in range(seq_len):
            # if k == 0:
            #     x = torch.empty(batch_size, 0, dtype=torch.long).to(self.device)
            # else:
            #     x = xs
            # tgt_mask = self.generate_square_subsequent_mask(generated.size(0), generated.size(1))
            cur_gen = generated if k == 0 else generated[:, k-1:k]
            logits, mems = self.forward(prompt_texts, cur_gen, m_lens, labels[:,k:k+1], mems=mems, is_generating=True)
            # pred_ids = pred_id[:,-1:]
            # filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            probs = F.softmax(logits[:,-1,:] / temperature, dim=-1)  # (b, seqlen, ntoken)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
            # pred_ids = torch.multinomial(probs, num_samples=1)  # (b, seqlen)
            dist = Categorical(probs)
            pred_ids = dist.sample()
            # if pred_ids == self.opt.num_tokens:
            #     pred_ids = 0
            pred_ids = pred_ids.unsqueeze(-1)
            
                
            # if k == 0:
            #     xs = pred_ids
            # else:
            # generated = torch.cat((generated, pred_ids), dim=1)
            # output = self.seqTransDecoder(x, memory=memory, tgt_mask=tgt_mask)[1:] #(seqlen, b, e)
            # logits = self.output_process(output) #(seqlen, b, e) -> (b, seqlen, n)
            # logits = logits[:, -1:, :]
            # filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            # pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1) # (b, seqlen)
            # x = self.embedding(pred_ids, torch.ones_like(pred_ids, dtype=torch.long), prefix_len=generated.size(1))
            # x = self.input_process(x)
            # # generated = torch.cat([generated, x], dim=1).clone()
            # segment_ids = torch.ones_like(generated)
            res_seq_ids.append(pred_ids)
            generated = torch.cat(res_seq_ids, dim=1)
            # if k == seq_len - 1:
            #     generated = generated[:, :-1]
        # print(f"motion res_seq_ids ========================+> {res_seq_ids}")
        motion_ids = generated.to(self.device)
        # non_pad_mask = lengths_to_mask(m_lens, motion_ids.size(1))
        # motion_ids = torch.where(non_pad_mask, motion_ids, 0)
        print(f"motion motion_ids ========================+> {motion_ids}\n labels========================+> {labels}")
        # gathered_ids = repeat(motion_ids.unsqueeze(-1), 'b n -> b n d', d=6)
        pred_motions = self.vq_model.forward_decoder(motion_ids.unsqueeze(-1))
        print(f"motion pred_motions ========================+> {pred_motions.shape}\n labels========================+> {labels.shape}")
        self.seqTransDecoderXL.reset_length(210, 210, 210)
        self.seqTransDecoderXL.train()
        
        return pred_motions