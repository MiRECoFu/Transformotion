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
        _num_tokens = opt.num_tokens + 1 # for motion pad and end
        print(f"opt.num tokens ====={opt.num_tokens}")
        self.num_tokens = _num_tokens
        # self.end_id = opt.num_tokens
        self.pad_id = opt.num_tokens
        self.opt = opt
        self.num_heads = num_heads
        self.prompt_drop_prob = prompt_drop_prob
        
        cutoffs, tie_projs = [], [False]
        
        self.decoder_xl_dim = 1024
        self.cond_emb = nn.Linear(clip_dim, self.decoder_xl_dim)
        self.head = nn.Linear(self.decoder_xl_dim, opt.num_tokens, bias=False)
        self.seqTransDecoderXL = TrmXLDecoder(self.num_tokens, num_layers, num_heads,
                            self.decoder_xl_dim, d_head=self.decoder_xl_dim // num_heads, d_inner=1024*4, dropout=dropout,
                            dropatt=dropout, tie_weight=True, 
                            d_embed=1024, div_val=1, 
                            tie_projs=tie_projs, pre_lnorm=False,
                            tgt_len=198, ext_len=198, mem_len=198,
                            cutoffs=cutoffs).to(self.device)
        
        self.apply(self.__init_weights)
        self.seqTransDecoderXL.apply(weights_init)
        self.seqTransDecoderXL.word_emb.apply(weights_init)
        
        self.mems = tuple()

        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)

       
    
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
        output =  self.seqTransDecoder(xseq)
        logits = self.head(output)
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
            labels = motion_ids
            motion_ids = motion_ids[:, :-1]
            # labels = motion_ids
            motion_ids = self.mask_motion_token(motion_ids)
        # print(f"prompt_logits=====>{prompt_logits.shape}")
        # mems = mems
        mems=tuple()
        ret, pred_hid = self.seqTransDecoderXL(is_generating, prompt_logits.unsqueeze(0), motion_ids.permute(1, 0), labels.permute(1, 0))
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


    # not used
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

        # print(f"mask====={mask}")
        # print(f"gen mask=======++>{mask}")
        return mask

    
    @torch.no_grad()
    @eval_decorator
    def generate(self, prompt_texts, m_lens, labels=None, temperature=0.6, tk=1, topk_filter_thres=0.9, cond_scale=3):
        self.eval()
        self.seqTransDecoderXL.eval()
        seq_len = max(m_lens).to(self.device)
        batch_size = len(m_lens)
    
        res_seq_ids = []
        mems = tuple()
        
        # segment_ids = torch.ones_like(generated)
        generated = torch.empty(batch_size, 0, dtype=torch.long).to(self.device)
        for k in range(seq_len):
            # if k == 0:
            #     x = torch.empty(batch_size, 0, dtype=torch.long).to(self.device)
            # else:
            #     x = xs
            # tgt_mask = self.generate_square_subsequent_mask(generated.size(0), generated.size(1))
            # cur_gen = generated if k == 0 else generated[:, k-1:k]
            logits, mems = self.forward(prompt_texts, generated, m_lens, labels=labels[:, k:k+1], mems=None, is_generating=True)
            # print(f"logits==========+>{logits.shape}")
            # pred_ids = pred_id[:,-1:]
            # filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            probs = F.softmax(logits[:,-1,:] / temperature, dim=-1)  # (b, seqlen, ntoken)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
            # pred_ids = torch.multinomial(probs, num_samples=1)  # (b, seqlen)
            dist = Categorical(probs)
            pred_ids = dist.sample()
            # if pred_ids == self.end_id:
            #     break
            # if pred_ids == self.opt.num_tokens:
            #     pred_ids = 0
            pred_ids = pred_ids.unsqueeze(-1)
            # print(f"pred_ids======={pred_ids}, \ngenerated========{generated}")

            res_seq_ids.append(pred_ids)
            generated = torch.cat(res_seq_ids, dim=1)
            # if k == seq_len - 1:
            #     generated = generated[:, :-1]
        # print(f"motion res_seq_ids ========================+> {res_seq_ids}")
        motion_ids = torch.cat(res_seq_ids, dim=1).to(self.device)
        print(f"motion motion_ids ========================+> {motion_ids}\n labels====> {labels}")
        # gathered_ids = repeat(motion_ids.unsqueeze(-1), 'b n -> b n d', d=6)
        pred_motions = self.vq_model.forward_decoder(motion_ids.unsqueeze(-1))
        # print(f"motion pred_motions ========================+> {pred_motions.shape}\n labels========================+> {labels.shape}")
        self.seqTransDecoderXL.reset_length(198, 198, 198)
        self.seqTransDecoderXL.train()
        
        return pred_motions