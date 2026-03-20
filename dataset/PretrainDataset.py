from torch.utils.data import Dataset
from datasets import load_dataset
import torch

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.samples = load_dataset('json', data_files=self.data_path, split='train') # 指定整个数据集都是训练集
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 因为要在首位添加开始标记<BOS>和结束标记<EOS>，所以最大长度需要减去2，如果文本太长会直接阶段，拿到分词后的token ids
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        # 开始添加开始标记<BOS>和结束标记<EOS>
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        
        # 开始使用填充标记<PAD>进行填充，让每个样本长度一致
        tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        # 将token转为tensor
        tokens = torch.tensor(tokens).long()
        
        # 获取label
        labels = tokens.clone()
        # 在label中将标记为<PAD>的部分设为-100，在计算交叉熵损失的时候就会自动忽略这些位置
        labels[tokens == self.tokenizer.pad_token_id] = -100
        
        #构建attention mask 填充<PAD>的地方不应该影响最终注意力分数的计算
        attention_mask = (tokens != self.tokenizer.pad_token_id).long()
        
        return tokens, labels, attention_mask
