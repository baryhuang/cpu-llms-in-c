"""Decoder-only Whisper graph shaped for RKLLM's custom-model converter.

The stored residual stream is centered across the hidden dimension.  That
makes RMSNorm exactly reproduce Whisper LayerNorm's variance term.  The
offline packer folds every LayerNorm beta into the following projection and
centers each residual-branch output projection.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_whisper_rkllm import WhisperRKLLMConfig


class WhisperRKLLMRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        source_dtype = hidden_states.dtype
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.variance_epsilon)
        return (normalized * self.weight.float()).to(source_dtype)


class WhisperRKLLMSelfAttention(nn.Module):
    def __init__(self, config: WhisperRKLLMConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        del attention_mask, position_ids, output_attentions, use_cache, kwargs
        batch, tokens, hidden = hidden_states.shape
        shape = (batch, tokens, self.num_heads, self.head_dim)
        q, k, v = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        value = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        value = value.transpose(1, 2).contiguous().view(batch, tokens, hidden)
        return self.o_proj(value), None, past_key_value


class WhisperRKLLMCrossAttention(nn.Module):
    def __init__(self, config: WhisperRKLLMConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.cross_q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.cross_o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_k: torch.Tensor,
        encoder_v: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        encoder_pos=None,
        **kwargs,
    ) -> torch.Tensor:
        del encoder_pos, kwargs
        batch, tokens, hidden = hidden_states.shape
        q = self.cross_q_proj(hidden_states)
        q = q.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        if encoder_k.ndim == 3:
            encoder_k = encoder_k.view(batch, -1, self.num_heads, self.head_dim)
            encoder_v = encoder_v.view(batch, -1, self.num_heads, self.head_dim)
        k = encoder_k.transpose(1, 2)
        v = encoder_v.transpose(1, 2)
        mask = None
        if encoder_mask is not None:
            mask = encoder_mask[:, None, None, :].to(torch.bool)
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        value = value.transpose(1, 2).contiguous().view(batch, tokens, hidden)
        return self.cross_o_proj(value)


class WhisperRKLLMMLP(nn.Module):
    def __init__(self, config: WhisperRKLLMConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return self.down_proj(F.gelu(self.up_proj(hidden_states)))


class WhisperRKLLMDecoderLayer(nn.Module):
    def __init__(self, config: WhisperRKLLMConfig):
        super().__init__()
        self.input_layernorm = WhisperRKLLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = WhisperRKLLMSelfAttention(config)
        self.cross_layernorm = WhisperRKLLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.cross_attn = WhisperRKLLMCrossAttention(config)
        self.post_attention_layernorm = WhisperRKLLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = WhisperRKLLMMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        encoder_mask: Optional[torch.Tensor] = None,
        encoder_k: Optional[torch.Tensor] = None,
        encoder_v: Optional[torch.Tensor] = None,
        encoder_pos=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        del kwargs
        attended, attention, present = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
        )
        hidden_states = hidden_states + attended
        if encoder_k is not None and encoder_v is not None:
            cross = self.cross_layernorm(hidden_states)
            hidden_states = hidden_states + self.cross_attn(
                cross, encoder_k, encoder_v, encoder_mask, encoder_pos)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention,)
        if use_cache:
            outputs += (present,)
        return outputs


class WhisperRKLLMPreTrainedModel(PreTrainedModel):
    config_class = WhisperRKLLMConfig
    base_model_prefix = "model"
    _no_split_modules = ["WhisperRKLLMDecoderLayer"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class WhisperRKLLMModel(WhisperRKLLMPreTrainedModel):
    def __init__(self, config: WhisperRKLLMConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [WhisperRKLLMDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = WhisperRKLLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_k_cache: Optional[torch.Tensor] = None,
        encoder_v_cache: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        del kwargs
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.embed_tokens(input_ids)
        batch, tokens, _ = inputs_embeds.shape
        # Learned Whisper positions are added by the native embedding callback.
        # Keeping an nn.Embedding here creates an unsupported, unmapped tensor
        # in RKLLM's custom-model graph.
        del position_ids
        hidden_states = inputs_embeds
        for layer_index, layer in enumerate(self.layers):
            layer_k = None if encoder_k_cache is None else encoder_k_cache[:, layer_index]
            layer_v = None if encoder_v_cache is None else encoder_v_cache[:, layer_index]
            hidden_states = layer(
                hidden_states,
                encoder_mask=encoder_mask,
                encoder_k=layer_k,
                encoder_v=layer_v,
            )[0]
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class WhisperRKLLMForCausalLM(WhisperRKLLMPreTrainedModel):
    def __init__(self, config: WhisperRKLLMConfig):
        super().__init__(config)
        self.model = WhisperRKLLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
