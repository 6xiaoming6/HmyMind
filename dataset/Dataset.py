from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch
from transformers import AutoTokenizer
import random


# 如果对话的开头没有系统提示词的话，以 add_system_ratio 的概率向开头随机添加一段系统提示词
# 让模型在微调时接触到多样化的系统角色设定，增强模型的鲁棒性和适应性
def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]

    if conversations and conversations[0].get('role') != 'system' and random.random() < add_system_ratio:
        conversations = [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations

    return conversations


# 以一定概率清除 prompt_content 中的 <think> 标签
def post_processing_chat(prompt_content, empty_think_ratio=0.5):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() < empty_think_ratio:
        prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = load_dataset('json', data_files=self.data_path, split='train')  # 指定整个数据集都是训练集

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 因为要在首位添加开始标记<BOS>和结束标记<EOS>，所以最大长度需要减去2，如果文本太长会直接阶段，拿到分词后的token ids
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2,
                                truncation=True).input_ids
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

        # 构建attention mask 填充<PAD>的地方不应该影响最终注意力分数的计算
        attention_mask = (tokens != self.tokenizer.pad_token_id).long()

        return tokens, labels, attention_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')

        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    # 利用 Hugging Face 分词器的 apply_chat_template 方法，将一组指定格式的对话消息转换为模型可理解的纯文本提示 prompt
    def create_chat_prompt(self, conversations):
        messages = conversations.copy()
        tools = conversations[0]['functions'] if (
                    conversations and conversations[0]['role'] == 'system' and conversations[0].get(
                'functions')) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 将格式化后的序列以文本形式返回，不进行token化
            add_generation_prompt=False,  # 设置为True会在对话的末尾添加例如<|im_start|>assistant的提示，告诉模型接下来需要回复消息
            tools=tools
        )

    # 只让模型对“助手（assistant）”的回复部分进行损失计算，而用户输入、系统提示等部分全部屏蔽（设为 -100）
    def generate_labels(self, input_ids):
        # 初始默认都设为 -100，只把参与计算的部分(模型回复的部分)设为对应的token_id
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i: i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)  # 模型回复消息的开始位置
                # 找到模型回复消息的结束位置(数组末尾和eos_token的地方)
                end = start
                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    else:
                        end += 1
                # [start, end + len(self.eos_id)]就是模型要回复的部分，只对这部分计算loss，同时也要保证输入的token序列不能超过最大长度
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                # 更新 i，接着找到下一个对话中的模型输出部分
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1

        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])  # 以一定的概率为该对话增加系统身份的提示词
        prompt = self.create_chat_prompt(conversations)  # 将对话转换为模板prompt
        prompt = post_processing_chat(prompt)
        # 将pompt转为token
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        # 长度不够的使用padding进行填充
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('../model')
    dataset = PretrainDataset(data_path='./data/pretrain_hq.jsonl', tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True)
    for (input_ids, labels, attention_mask) in loader:
        print(input_ids)
        print(labels)
        input("按回车继续\n")
