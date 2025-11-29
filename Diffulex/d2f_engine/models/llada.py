import os
import torch
import torch.nn as nn
import torch.distributed as dist

from d2f_engine.layers.activation import SiluAndMul
from d2f_engine.layers.attention.attention_v5 import Attention
from d2f_engine.layers.layernorm import RMSNorm
from d2f_engine.layers.linear import RowParallelLinear, ColumnParallelLinear
from d2f_engine.layers.rotary_embedding import get_rope
from d2f_engine.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from d2f_engine.models.config.llada.configuration_llada import LLaDAConfig


if os.environ.get("TRITON_INTERPRET", None) == "1":
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = True
    torch.backends.optimized_mode = False


class LLaDARMSNorm(RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)


class LLaDAAttention(nn.Module):
    """LLaDA attention."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=qkv_bias,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=qkv_bias,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            "diffusion_lm",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        import logging
        logger = logging.getLogger('d2f_engine.models.llada')

        logger.debug(f"  Attention input: hidden_states has_nan={torch.isnan(hidden_states).any()}, mean={hidden_states.mean():.4f}")

        # Debug: Check projection weights and biases
        logger.debug(f"  q_proj.weight: has_nan={torch.isnan(self.q_proj.weight).any()}, has_inf={torch.isinf(self.q_proj.weight).any()}")
        if self.q_proj.bias is not None:
            logger.debug(f"  q_proj.bias: has_nan={torch.isnan(self.q_proj.bias).any()}, has_inf={torch.isinf(self.q_proj.bias).any()}, mean={self.q_proj.bias.mean():.6f}")
        else:
            logger.debug(f"  q_proj.bias: None")

        q = self.q_proj(hidden_states)
        logger.debug(f"  After q_proj: has_nan={torch.isnan(q).any()}, mean={q.mean() if not torch.isnan(q).any() else float('nan')}")

        k = self.k_proj(hidden_states)
        logger.debug(f"  After k_proj: has_nan={torch.isnan(k).any()}, mean={k.mean() if not torch.isnan(k).any() else float('nan')}")

        v = self.v_proj(hidden_states)
        logger.debug(f"  After v_proj: has_nan={torch.isnan(v).any()}, mean={v.mean() if not torch.isnan(v).any() else float('nan')}")
        
        # Handle both batched (batch_size, seq_len, hidden_size) and unbatched (seq_len, hidden_size) inputs
        # This is needed because vLLM may pass different shapes during warmup vs actual generation
        has_batch_dim = q.dim() == 3
        if has_batch_dim:
            batch_size, seq_len, hidden_size = q.shape
            q_flat = q.view(-1, hidden_size)  # (batch * seq_len, hidden_size)
            k_flat = k.view(-1, hidden_size)  # (batch * seq_len, hidden_size)
            positions_flat = positions.view(-1)  # Flatten positions too
        else:
            # Unbatched: (seq_len, hidden_size)
            seq_len, hidden_size = q.shape
            batch_size = 1
            q_flat = q  # Already (seq_len, hidden_size)
            k_flat = k
            positions_flat = positions.view(-1) if positions.dim() > 1 else positions
        
        q_flat, k_flat = self.rotary_emb(positions_flat, q_flat, k_flat)
        
        # Prepare v for attention (same shape as q_flat and k_flat)
        if has_batch_dim:
            v_flat = v.view(-1, hidden_size)
        else:
            v_flat = v
        
        # Attention layer expects (seq_len, hidden_size) or (batch * seq_len, hidden_size)
        # It returns (seq_len, hidden_size) for diffusion_lm in prefill
        o_attn = self.attn(q_flat, k_flat, v_flat, mask)
        
        # Ensure output has the same shape as input
        if has_batch_dim:
            # Input was (batch_size, seq_len, hidden_size)
            if o_attn.shape[0] == batch_size * seq_len:
                # Output is (batch * seq_len, hidden_size) - reshape back
                o_attn = o_attn.view(batch_size, seq_len, -1)
            elif o_attn.shape[0] == seq_len:
                # Output is (seq_len, hidden_size) - add batch dimension
                o_attn = o_attn.unsqueeze(0)
            # else: already correct shape
        else:
            # Input was (seq_len, hidden_size) - output should match
            if o_attn.shape[0] != seq_len:
                # If output has batch dimension, remove it
                o_attn = o_attn.view(seq_len, -1)
        
        output = self.o_proj(o_attn)
        return output


class LLaDAMLP(nn.Module):
    """LLaDA MLP."""
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.act_fn(torch.cat([gate, up], dim=-1))
        x = self.down_proj(x)
        return x


class LLaDABlock(nn.Module):
    """LLaDA transformer block."""
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.self_attn = LLaDAAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.n_kv_heads,
            max_position=config.max_sequence_length,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'include_qkv_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = LLaDAMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.mlp_hidden_size,
            hidden_act=config.activation_type,
        )
        self.input_layernorm = LLaDARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LLaDARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import logging
        logger = logging.getLogger('d2f_engine.models.llada')

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            logger.debug(f"  After input_layernorm: has_nan={torch.isnan(hidden_states).any()}, mean={hidden_states.mean():.4f}")
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            logger.debug(f"  After input_layernorm (with residual): has_nan={torch.isnan(hidden_states).any()}, mean={hidden_states.mean():.4f}")

        hidden_states = self.self_attn(positions, hidden_states, mask)
        has_nan = torch.isnan(hidden_states).any()
        logger.debug(f"  After self_attn: has_nan={has_nan}")
        if not has_nan:
            logger.debug(f"  After self_attn: mean={hidden_states.mean():.4f}")

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        has_nan = torch.isnan(hidden_states).any()
        logger.debug(f"  After post_attention_layernorm: has_nan={has_nan}")
        if not has_nan:
            logger.debug(f"  After post_attention_layernorm: mean={hidden_states.mean():.4f}")

        hidden_states = self.mlp(hidden_states)
        has_nan = torch.isnan(hidden_states).any()
        logger.debug(f"  After mlp: has_nan={has_nan}")
        if not has_nan:
            logger.debug(f"  After mlp: mean={hidden_states.mean():.4f}")

        return hidden_states, residual


class LLaDAModel(nn.Module):
    """LLaDA backbone."""
    def __init__(
        self,
        config: LLaDAConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=VocabParallelEmbedding(
                    config.embedding_size or config.vocab_size, config.d_model
                ),
                emb_drop=nn.Dropout(config.embedding_dropout),
                ln_f=LLaDARMSNorm(config.hidden_size, config.rms_norm_eps)
            )
        )
        
        blocks = [LLaDABlock(config) for _ in range(config.n_layers)]
        self.transformer.update({"blocks": nn.ModuleList(blocks)})
        
        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            ) 
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        import logging
        logger = logging.getLogger('d2f_engine.models.llada')

        # Debug: Check embeddings
        embeddings = self.transformer.wte(input_ids)
        logger.debug(f"Embeddings: shape={embeddings.shape}, has_nan={torch.isnan(embeddings).any()}, mean={embeddings.mean():.4f}")

        hidden_states = self.transformer.emb_drop(embeddings)
        logger.debug(f"After emb_drop: has_nan={torch.isnan(hidden_states).any()}, mean={hidden_states.mean():.4f}")

        residual = None
        for block_idx, block in enumerate(self.transformer.blocks):
            hidden_states, residual = block(positions, hidden_states, residual, mask)
            # Debug first 3 blocks
            if block_idx < 3:
                logger.debug(f"After block {block_idx}: has_nan={torch.isnan(hidden_states).any()}, mean={hidden_states.mean():.4f}")

        hidden_states, _ = self.transformer.ln_f(hidden_states, residual)
        logger.debug(f"After ln_f: has_nan={torch.isnan(hidden_states).any()}, mean={hidden_states.mean():.4f}")

        return hidden_states


class LLaDAForDiffusionLM(nn.Module):
    """LLaDA with LM head."""
    packed_modules_mapping = {
        "q_proj": ("self_attn.q_proj", None),
        "k_proj": ("self_attn.k_proj", None),
        "v_proj": ("self_attn.v_proj", None),
        "attn_out": ("self_attn.o_proj", None),
        "attn_norm": ("input_layernorm", None),
        "ff_norm": ("post_attention_layernorm", None),
        
        "ff_proj": ("mlp.gate_proj", None),
        "up_proj": ("mlp.up_proj", None),
        "ff_out": ("mlp.down_proj", None),
        
        "transformer.ff_out": ("lm_head", None)
    }
    
    def __init__(
        self,
        config: LLaDAConfig,
    ) -> None:
        super().__init__()
        self.model = LLaDAModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, model_type='diffusion_lm')
        if getattr(config, 'weight_tying', False):
            self.lm_head.weight.data = self.model.transformer.wte.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, mask)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
