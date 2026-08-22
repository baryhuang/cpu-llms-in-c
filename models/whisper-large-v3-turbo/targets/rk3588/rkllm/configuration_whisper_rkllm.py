"""Configuration for the decoder-only RKLLM Whisper bridge."""

from transformers import PretrainedConfig


class WhisperRKLLMConfig(PretrainedConfig):
    model_type = "whisper_rkllm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=51866,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=4,
        num_attention_heads=20,
        num_key_value_heads=20,
        max_position_embeddings=448,
        hidden_act="gelu",
        rms_norm_eps=1.0e-5,
        attention_bias=True,
        no_rope_layer_interval=1,
        rope_theta=10000.0,
        bos_token_id=50257,
        eos_token_id=50257,
        pad_token_id=50257,
        **kwargs,
    ):
        kwargs.pop("tie_word_embeddings", None)
        kwargs.pop("is_decoder", None)
        kwargs.pop("is_encoder_decoder", None)
        kwargs.pop("use_cache", None)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        # RKLLM uses this metadata to suppress RoPE for every decoder layer.
        self.no_rope_layer_interval = no_rope_layer_interval
        # The custom converter still expects base RoPE metadata even when all
        # layers are disabled; runtime 1.3.0's unused-copy bug is handled by
        # the separately pinned compatibility utility.
        self.rope_theta = rope_theta
        self.use_cache = True
        self.pretraining_tp = 1
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=False,
            is_decoder=True,
            is_encoder_decoder=False,
            **kwargs,
        )
