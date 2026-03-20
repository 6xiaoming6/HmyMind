import math
from typing import List, Optional, Tuple, Union
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn
import torch.nn.functional as F

#创建了一个模型配置类来管理所有的模型参数和配置
class HmyMindConfig(PretrainedConfig):
    model_type = "hmymind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = 'silu',
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000.0,
        inference_rope_scaling: bool = False,
        flash_attn: bool = True,
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(self.dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)  # 1 / 根号 x
        return self.weight * x


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None
):
    #求出每个位置的频率，频率是根据位置和维度计算出来的 求出 θ
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    attn_factor = rope_scaling["attention_factor"] if rope_scaling is not None and "attention_factor" in rope_scaling else 1.0

    #如果开启了旋转位置编码的缩放，就是用yarn方法对频率进行缩放
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0)
        )
        #输入长度大于原始最大长度时进行缩放（缩放因子默认是16）
        if end / orig_max > 1.0:
            #输入比值计算对应的维度
            inv_dim = lambda r: dim * math.log(orig_max / (2 * math.pi * r)) / (2 * math.log(rope_base))
            #维度 <= low的都是高频（维度越小频率越高），不需要缩放；维度 >= high的都是低频（维度越大频率越低），需要缩放；中间的维度是过渡频率
            low, high = math.floor(max(inv_dim(beta_fast), 0)), math.ceil(min(inv_dim(beta_slow), dim // 2 - 1))
            #根据维度对混合系数进行分段，如果维度小于等于low，混合系数为0；如果维度大于等于high，混合系数为1；如果维度在low和high之间，混合系数根据线性函数进行计算
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            #应用缩放(0不缩放，1完全线性缩放，中间的数字进行线性插值适应性缩放)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device).float()  #长度为end
    freqs = torch.outer(t, freqs)  # 行数为end，列数为dim//2，freqs[i]表示第i个向量所对应的频率，求出来 m * θ
    #计算每个角度对应的正弦和余弦值，因为两个相邻位置的频率相同，所以直接复制一份拼接起来即可
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat([-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]], dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


#将键（key）和值（value）张量的头数维度重复，使其与查询（query）的头数维度一致
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(
            bs, slen, num_key_value_heads * n_rep, head_dim
        )
    )


class Attention(nn.Module):
    def __init__(self, args: HmyMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads  #这里的kv_heads可以理解为一共分了几组kv
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  #n_rep表示每个kv组需要被多少q共享
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  #接受计算的cos和sin
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  #接受之前计算的key和value
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim)  #(batch_size, seq_len, num_attention_heads, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)  #(batch_size, seq_len, num_key_value_heads, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)  #(batch_size, seq_len, num_key_value_heads, head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, position_embeddings[0], position_embeddings[1])  #对q和k进行旋转位置编码

        #使用kv cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        #xq变为(batch_size, num_attention_heads, seq_len, head_dim)
        #xk和xv变为(batch_size, num_attention_heads, seq_len, head_dim)
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        #注意力计算(目前还不用flash)
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_attention_heads, seq_len, seq_len)
            mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(scores.device)  #(seq_len, seq_len)
            scores[:, :, :, -seq_len:] += mask

            #通常用于处理 padding token 或某些需要屏蔽的特殊情况，<pad>的位置会被标记为0
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = torch.softmax(scores, dim=-1)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)  #变回(batch_size, seq_len, num_attention_heads * head_dim)
        output = self.resid_dropout(self.o_proj(output))  #通过输出层投影回(batch_size, seq_len, hidden_size)

        return (output, past_kv)


class FeedForward(nn.Module):
    def __init__(self, config: HmyMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)  #确保intermediate_size是一个整数
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]  #SiLU激活函数

    def forward(self, x: torch.Tensor):
        return self.dropout(
            self.down_proj(
                self.up_proj(x) * self.act_fn(self.gate_proj(x))
            )
        )


class MOEGate(nn.Module):
    def __init__(self, config: HmyMindConfig):
        super().__init__()
        self.config = config
        self.topk = config.num_experts_per_tok  #每个token选择的专家数，也就是topk的值
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func  # 路由计算得分的函数
        self.alpha = config.aux_loss_alpha  # 辅助损失的权重
        self.seq_aux = config.seq_aux  # 是否在序列级别计算辅助损失，即是在一个序列内统计token分配次数求损失再求整个batch的平均，还是在整个batch内统计token分配次数求损失
        self.norm_topk_prob = config.norm_topk_prob  # 是否将激活的专家的权重再归一化，比如激活的两个专家权重为(0.2, 0.4)，归一化后就变为(0.33, 0.67))
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.reset_parameters()  # 初始化模型参数

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape  # (batch_size, seq_len, hidden_dim)
        hidden_states = hidden_states.view(-1, hidden_dim)  # (batch_size * seq_len, hidden_dim)
        logits = F.linear(hidden_states, self.weight)  # (batch_size * seq_len, n_routed_experts),计算出每个token分配到每个专家logits

        # 对logits进行softmax得到每个token分配到每个专家的概率
        if self.scoring_func == 'softmax':
            scores = F.softmax(logits, dim=-1)  # (batch_size * seq_len, n_routed_experts)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weights, topk_ids = torch.topk(scores, self.topk, dim=-1, sorted=False)  # (batch_size * seq_len, self.topk)

        # topk激活的专家得分归一化
        if self.topk > 1 and self.norm_topk_prob:
            topk_weights_sum = torch.sum(topk_weights, dim=-1, keepdim=True) + 1e-20  # (batch_size * seq_len, 1), 加上1e-20防止除零
            topk_weights /= topk_weights_sum

        # 计算辅助损失
        # self.training是 nn.Module 内置的属性，显示当前是否处于训练状态，由 model.train()和 model.eval()控制
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores  # (batch_size * seq_len, n_routed_experts)
            topk_ids_for_aux = topk_ids.view(batch_size, -1)  # (batch_size, seq_len * topk)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(batch_size, seq_len, -1)  # (batch_size, seq_len, n_routed_experts)
                ce = torch.zeros(batch_size, self.n_routed_experts, device=hidden_states.device)  # (batch_size, n_routed_experts)
                ce.scatter_add_(1, topk_ids_for_aux, torch.ones_like(topk_ids_for_aux, device=hidden_states.device, dtype=ce.dtype))  # 统计每个专家被分配到的token数量
                ce.div_(seq_len * self.topk / self.n_routed_experts)  # seq_len * self.topk / self.n_routed_experts 是每个专家期望分配到的topken数量，实际token / 期望token数得到相对负载
                # 计算辅助损失 (batch_size, n_routed_experts) --sum--> (batch_size) --mean--> (1)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                #得到的矩阵中每一行对应一个选中的专家，该专家的位置为 1，其余为 0。
                mask_ce = F.one_hot(topk_ids_for_aux.view(-1), num_classes=self.n_routed_experts)  # (batch_size * seq_len * topk, n_routed_experts)
                ce = mask_ce.float().mean(dim=0)  #(n_routed_experts)，统计一个batch中每个专家被分配到的token的比例
                Pi = scores_for_aux.mean(dim=0)  #(n_routed_experts)，统计一个batch中每个专家的平均分配概率
                fi = ce * self.n_routed_experts
                aux_loss = (fi * Pi).sum() * self.alpha
        else:
            # 在 scores 所在的设备上创建一个形状为 (1,) 的全零张量，数据类型与 scores 相同。最后在移除这个维度，变成一个标量张量
            aux_loss = scores.new_zeros(1).squeeze()

        return topk_ids, topk_weights, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: HmyMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([FeedForward(config) for i in range(config.n_routed_experts)])
        self.gate = MOEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([FeedForward(config) for i in range(config.n_shared_experts)])

    def forward(self, x: torch.Tensor):
        identity = x  # (batch_size, seq_len, hidden_dim)
        origin_shape = x.shape
        batch_size, seq_lenm, hidden_dim = x.shape
        # 路由网络选择专家
        topk_ids, topk_weights, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])  # (bsz*seq_len, hidden)
        flat_topk_idx = topk_ids.view(-1)  # (bsz*seq_len*topk)

        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)  # (bsz*seq_len*topk, hidden)
            y = torch.empty_like(x, dtype=x.dtype)  # (bsz*seq_len*topk, hidden)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            y = (y.view(*topk_weights.shape, -1) * topk_weights.unsqueeze(-1)).sum(dim=1)
            y = y.view(*origin_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weights.view(-1, 1)).view(*origin_shape)

        # 累加上共享专家的输出
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)  # 计算每个专家分配到的token数目
        token_idxs = idxs // self.config.num_experts_per_tok  # 计算出每个位置所属的token的是哪一个
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25, 4, 5, 6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4, 5, 6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]  # 拿到第i个专家网络
            exp_token_idx = token_idxs[start_idx:end_idx]  # 拿到分配给这个专家的token的位置
            expert_tokens = x[exp_token_idx]  # 根据位置从数据中去除对应的token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)  # 拿到该专家处理的token后输入到专家网络中计算得到每个token的输出
            # 每个专家对应token输出的结果加权求和
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class HmyMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: HmyMindConfig):
        super().__init__()
        self.layer_id = layer_id
        self.attn = Attention(config)
        self.ffn = FeedForward(config) if not config.use_moe else MOEFeedForward(config)  # 如果use_moe设为True，就把FFN替换为MOE层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor = None
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.attn(hidden_states, position_embeddings, past_key_value, use_cache, attention_mask)
        hidden_states += residual
        hidden_states = hidden_states + self.ffn(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


#处理输入的token_ids，经过嵌入层后进入模型计算，得到归一化后的输出hidden_states
class HmyMindModel(nn.Module):
    def __init__(self, config: HmyMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([HmyMindBlock(i, config) for i in range(config.num_hidden_layers)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #预计算旋转位置编码的频率，存储起来方便使用
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs
    ):
        batch_size, seq_len = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        # past_key_values 是一个列表，长度为层数。存储的是每一层历史缓存的key和value，列表中的元素是一个元组(key, value)
        # 元组内的kv的形状都是 (batch, past_len, num_key_value_heads, head_dim)
        # 所以去除历史缓存的长度，从past_len位置开始接着计算
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 根据输入的id经过嵌入层转为向量
        hidden_states = self.dropout(self.embedding(input_ids))
        position_embeddings = (
            self.freqs_cos[start_pos: start_pos + seq_len],
            self.freqs_sin[start_pos: start_pos + seq_len]
        )

        presents = []
        for layer_id, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # 将嵌入层输出的结果输入到每一层中计算，当前层的输出又会作为下一层的输入
            hidden_states, present_key_value = layer(hidden_states, position_embeddings, past_key_value, use_cache, attention_mask)
            # 如果use_cache == True，计算注意力的时候就会将当前新的key和value与历史缓存进行拼接并返回
            presents.append(present_key_value)

        # 对最后一层的输出进行归一化处理
        hidden_states = self.norm(hidden_states)

        # 如果使用了moe层，还需要返回额外的辅助损失
        aux_loss = sum([l.ffn.aux_loss for l in self.layers if isinstance(l.ffn, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())

        return hidden_states, presents, aux_loss


#增加模型最后的输出头，可直接用于训练
class HmyMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = HmyMindConfig

    def __init__(self, config: HmyMindConfig = None):
        self.config = config or HmyMindConfig()
        super().__init__(config=self.config)
        self.model = HmyMindModel(self.config)
        #模型的输出头，将隐藏层的hidden_size映射会vocab_size,最后通过softmax得到每个token的概率分布
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 二者的权重形状均为[vocab_size, hidden_size]，二者共享权重，减少了模型参数量同时可以让输入和输出的词向量空间保持一致
        self.model.embedding.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args
    ):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **args)
        # 一般只需要拿出模型输出的最后一个token即可，logits_to_keep可以控制拿到多少最后的输出结果
        slice_indeces = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indeces, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 先对shift_logits使用softmax得到每个token的概率分布，再计算交叉熵损失，-100表示在计算时忽略值为-100的位置，那些通常是padding
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output


if __name__ == "__main__":
    pass