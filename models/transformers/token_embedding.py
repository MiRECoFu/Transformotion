import torch
from models.vq.model import RVQVAE
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import os
# Set TOKENIZERS_PARALLELISM environment variable to disable parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MixEmbedding:
    def __init__(self, vq_model: RVQVAE, device, text_max_len=512) -> None:
        # RVQ model to quntize human motions to tokens
        self.vq_model = vq_model
        self.device = device
        self.vq_model.to(self.device)
        self.text_max_len = text_max_len
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        # 查看 [UNK] token 和它的 ID
        self.unk_token = self.tokenizer.unk_token
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token

        self.unk_token_id = self.tokenizer.convert_tokens_to_ids(self.unk_token)
        # 添加 special token
        # special_tokens_dict = {'additional_special_tokens': ['[SOM]', '[EOM]']}
        # self.num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        # 加载预训练的模型
        self.model = AutoModel.from_pretrained('bert-base-cased')
        self.model.to(self.device)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.vocab_size = len(self.tokenizer)
        # 增加 tokenizer 中的特殊 token 数
        # self.model.resize_token_embeddings(len(self.tokenizer))

        # # 取得新增 special token 的嵌入向量
        # # self.som_token_id = self.tokenizer.convert_tokens_to_ids('[SOM]')
        # # self.eom_token_id = self.tokenizer.convert_tokens_to_ids('[EOM]')
        # self.cls_token_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        # self.sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        # self.pad_token_id = self.tokenizer.pad_token_id
        # self.som_token_embedding = self.model.embeddings.word_embeddings.weight[self.som_token_id]
        # self.eom_token_embedding = self.model.embeddings.word_embeddings.weight[self.eom_token_id]
        
        # embedding_matrix = self.model.embeddings.word_embeddings.weight.data
        # vocab_size, embedding_dim = embedding_matrix.shape
        # self.init_vocab_size = vocab_size
        # codebook = self.vq_model.quantizer.codebooks[0]
        # codebook_size, codebook_dim = codebook.shape
        # new_vocab_size = vocab_size + codebook_size
        # new_embedding_matrix = torch.zeros(new_vocab_size, embedding_dim)
        # new_embedding_matrix[:vocab_size, :] = embedding_matrix
        
        # new_embedding_matrix[vocab_size:, :] = codebook
        # self.model.embeddings.word_embeddings.weight = torch.nn.Parameter(new_embedding_matrix)
        # self.model.resize_token_embeddings(new_vocab_size)
        # self.model.embeddings.word_embeddings.requires_grad_(False)
        # self.vocab_size = new_vocab_size

    def get_vocab_size(self):
        return self.vocab_size

     # 将 [PAD] token 移到最后
    def move_pad_to_end(self, input_id):
        pad_mask = input_id == self.pad_token_id
        non_pad_tokens = input_id[~pad_mask]
        pad_tokens = input_id[pad_mask]
        return torch.cat([non_pad_tokens, pad_tokens])
    
    # texts: []
    def mix_embedding_ids(self, texts, motions):
        self.vq_model.eval()
        self.model.eval()
        # texts = texts.to(self.device)
        _motion_ids, _motion_emb = self.vq_model.encode(motions)
        _motion_ids = _motion_ids[..., 0] #[bs, seqlen]
        # mapping_table = torch.arange(self.init_vocab_size, self.vocab_size + 768)
        som_token_ids = torch.full((len(texts), 1), self.pad_token_id, device=self.device)
        motion_ids = torch.cat([som_token_ids, _motion_ids], dim=1).to(self.device) 
        batch_texts = texts
        encoded_inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_inputs['input_ids'].to(self.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        return motion_ids, input_ids, last_hidden_states, _motion_ids
        # tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        # batch_size = input_ids.size(0)
        # som_token_ids = torch.full((batch_size, 1), self.som_token_id, device=self.device)
        # eom_token_ids = torch.full((batch_size, 1), self.eom_token_id, device=self.device)
        # mixed_input_ids = torch.cat([input_ids, som_token_ids, motion_ids, eom_token_ids], dim=1)  
        # format_motion_ids = torch.cat([som_token_ids, _motion_ids, eom_token_ids], dim=1)
        # mixed_input_ids = torch.stack([self.move_pad_to_end(ids) for ids in mixed_input_ids])
        # Get the index of [SOM] token in the mixed_input_ids
        # som_token_index = (mixed_input_ids == self.som_token_id).nonzero(as_tuple=False)
        # som_token_index = som_token_index[:, 1].tolist()  # Extracting the index of [SOM] token
        # print(f"som_token_index=============================+> {som_token_index} \n input_ids size============+> {input_ids.size(1)}")
         # Generate segment_ids
        # segment_ids = torch.zeros_like(mixed_input_ids)
        # for idx, som_idx in enumerate(som_token_index):
        #     segment_ids[idx, som_idx:] = 1  # Set the segment_ids after [SOM] token to 1
        # return mixed_input_ids, som_token_index, format_motion_ids, segment_ids, input_ids

    def encode_text(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        return input_ids, last_hidden_states
        # som_token_ids = torch.full((batch_size, 1), self.som_token_id, device=self.device)
        # mixed_input_ids = torch.cat([input_ids], dim=1) 
        # # mixed_input_ids = torch.stack([self.move_pad_to_end(ids) for ids in mixed_input_ids])
        # # Get the index of [SOM] token in the mixed_input_ids
        # som_token_index = (input_ids == self.som_token_id).nonzero(as_tuple=False)
        # som_token_index = som_token_index[:, 1].tolist()  # Extracting the index of [SOM] token
        # # print(f"som_token_index=============================+> {som_token_index} \n input_ids size============+> {input_ids.size(1)}")
        #  # Generate segment_ids
        # segment_ids = torch.zeros_like(mixed_input_ids)
        # # for idx, som_idx in enumerate(som_token_index):
        # #     segment_ids[idx, som_idx:] = 1  # Set the segment_ids after [SOM] token to 1
        # print(f"mixed_input_ids shape:{mixed_input_ids.shape}, segment_ids: {segment_ids.shape}")
        # return mixed_input_ids, segment_ids
    
    def get_code_idx_from_token_ids(self, token_ids: torch.Tensor):
        is_in_codebook = (token_ids >= self.init_vocab_size) & (token_ids < self.vocab_size)
        # 获取在 codebook 中的 token ids
        codebook_token_ids = token_ids * is_in_codebook

        # 获取不在 codebook 中的 token ids，并设置为 -1 表示无效
        original_token_ids = token_ids * ~is_in_codebook + (-1) * is_in_codebook

        # 获取 code_idx，确保形状与 token_ids 相同
        code_idx = torch.where(is_in_codebook, token_ids - self.init_vocab_size, torch.tensor(-1).to(self.device))
        return code_idx




