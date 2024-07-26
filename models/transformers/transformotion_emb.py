import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from transformers import BertModel

class SequenceEmbeddingModel(torch.nn.Module):
    def __init__(self, embed_dim):
        super(SequenceEmbeddingModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.fc = torch.nn.Linear(768, embed_dim)  # 768是BERT的隐藏层大小

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 使用[CLS] token的输出
        embedding = self.fc(cls_output)
        return embedding