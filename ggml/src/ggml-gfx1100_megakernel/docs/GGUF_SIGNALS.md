# GGUF Signals for Kernel Composition

Every signal from GGUF that can inform the 6 meta-steps of the composition system.
Organized by which meta-step(s) each signal feeds into.

## 1. Architecture Identity

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `general.architecture` | string | DETECT | Architecture family (llama, gemma2, qwen2, phi3, mamba, rwkv6, t5, etc.) |
| `general.file_type` | u32 | COMPILE | Quantization level — affects which matvec kernels to include |
| `general.type` | string | DETECT | "model" vs "adapter" vs "vocab" — skip non-model files |

## 2. Core Dimensions (buffer sizing, grid dims, constexpr baking)

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `{arch}.embedding_length` | u32 | COMPILE, TUNE | Hidden size H — all buffer sizes, matvec dims |
| `{arch}.feed_forward_length` | u32 | COMPILE, TUNE | FFN intermediate size — MLP buffer, matvec dims |
| `{arch}.block_count` | u32 | COMPOSE | Number of layers — loop count, total launch count |
| `{arch}.vocab_size` | u32 | COMPILE | LM head output dim, logit buffer size |
| `{arch}.context_length` | u32 | TUNE | Max KV cache allocation, attention loop bound |
| `{arch}.embedding_length_out` | u32 | COMPILE | Output embedding dim (if different from input) |
| `{arch}.embedding_length_per_layer_input` | arr | COMPOSE | Per-layer input dim (OpenELM variable width) |

## 3. Attention Configuration

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `{arch}.attention.head_count` | u32 | TUNE | Q heads → parallel blocks (pb), CU occupancy |
| `{arch}.attention.head_count_kv` | u32 | TUNE | KV heads → GQA ratio, cache size |
| `{arch}.attention.key_length` | u32 | COMPILE | Head dim D → VEC vs WMMA kernel selection |
| `{arch}.attention.value_length` | u32 | COMPILE | Value dim (if different from key) |
| `{arch}.attention.key_length_swa` | u32 | COMPOSE | SWA-specific head dim (if different) |
| `{arch}.attention.value_length_swa` | u32 | COMPOSE | SWA-specific value dim |
| `{arch}.attention.causal` | bool | COMPOSE | Causal vs bidirectional attention kernel |
| `{arch}.attention.sliding_window` | u32 | COMPOSE, TUNE | SWA window size → per-layer cache strategy |
| `{arch}.attention.sliding_window_pattern` | str | COMPOSE | Which layers use SWA (alternating, etc.) |
| `{arch}.attention.scale` | f32 | COMPILE | Custom attention scale (non-1/sqrt(d)) |
| `{arch}.attention.clamp_kqv` | f32 | COMPOSE | QKV clamping (StarCoder) — extra op step |
| `{arch}.attention.max_alibi_bias` | f32 | COMPOSE | ALiBi positional encoding → different attn kernel |
| `{arch}.attention.group_norm_epsilon` | f32 | COMPOSE | Group norm in attention (Jina-BERT) |
| `{arch}.attention.group_norm_groups` | u32 | COMPOSE | Group norm groups |
| `{arch}.attention.shared_kv_layers` | u32 | COMPOSE | Cross-layer KV sharing (e.g., SmolLM3) |
| `{arch}.attention.output_scale` | f32 | COMPOSE | O-proj output scaling |
| `{arch}.attention.temperature_length` | u32 | COMPOSE | Dynamic temperature attention |
| `{arch}.attention.temperature_scale` | f32 | COMPOSE | Dynamic temperature scaling |
| `{arch}.attn_logit_softcapping` | f32 | COMPILE | Attention softcap (Gemma2/3/4) → kernel variant |
| `{arch}.final_logit_softcapping` | f32 | COMPILE | Final logit softcap → extra post-step |
| `{arch}.full_attention_interval` | u32 | COMPOSE | iSWA layer pattern (every Nth layer is full) |

## 4. RoPE Configuration

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `{arch}.rope.freq_base` | f32 | COMPILE | Base theta for RoPE |
| `{arch}.rope.freq_base_swa` | f32 | COMPILE | SWA-specific theta (Gemma3/4) |
| `{arch}.rope.dimension_count` | u32 | COMPILE | How many dims get RoPE rotation |
| `{arch}.rope.dimension_count_swa` | u32 | COMPILE | SWA-specific rope dim |
| `{arch}.rope.dimension_sections` | arr | COMPILE | Multi-RoPE sections (Qwen2VL/3VL) |
| `{arch}.rope.scaling.type` | u32 | COMPOSE | NONE/NORM/NEOX/MULTI → RoPE kernel variant |
| `{arch}.rope.scaling.factor` | f32 | COMPILE | Linear scaling factor |
| `{arch}.rope.scaling.attn_factor` | f32 | COMPILE | YaRN attention factor (mscale) |
| `{arch}.rope.scaling.yarn_ext_factor` | f32 | COMPILE | YaRN extrapolation factor |
| `{arch}.rope.scaling.yarn_attn_factor` | f32 | COMPILE | YaRN attention factor |
| `{arch}.rope.scaling.yarn_beta_fast` | f32 | COMPILE | YaRN frequency band boundary |
| `{arch}.rope.scaling.yarn_beta_slow` | f32 | COMPILE | YaRN frequency band boundary |
| `{arch}.rope.scaling.yarn_log_multiplier` | f32 | COMPILE | YaRN log multiplier |
| `{arch}.rope.scaling.original_context_length` | u32 | COMPILE | Original training context (for YaRN) |
| `{arch}.rope.scaling.finetuned` | bool | COMPILE | Whether RoPE was finetuned |
| `{arch}.rope.scale_linear` | f32 | COMPILE | Linear rope scaling |
| `rope_freqs.weight` (tensor) | f16 | COMPILE | Per-dim frequency factors (Llama 3 long RoPE) |

## 5. Normalization

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `{arch}.attention.layer_norm_rms_epsilon` | f32 | COMPILE | RMSNorm epsilon |
| `{arch}.attention.layer_norm_epsilon` | f32 | COMPILE | LayerNorm epsilon |
| `{arch}.swin_norm` | bool | COMPOSE | Swin-style norm placement (post- vs pre-) |
| `{arch}.embedding_scale` | f32 | COMPOSE | Scale embeddings (Gemma sqrt(H)) |
| `{arch}.residual_scale` | f32 | COMPOSE | Residual connection scaling (DeepSeek) |
| `{arch}.rescale_every_n_layers` | u32 | COMPOSE | Periodic rescaling (RWKV) |
| `{arch}.logit_scale` | f32 | COMPOSE | Final logit scaling |

## 6. FFN / MLP Configuration

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `{arch}.feed_forward_length` | u32 | COMPILE | FFN intermediate size |
| `{arch}.expert_count` | u32 | COMPOSE | MoE: number of experts |
| `{arch}.expert_used_count` | u32 | COMPOSE | MoE: top-K routing |
| `{arch}.expert_feed_forward_length` | u32 | COMPILE | MoE: per-expert FFN size |
| `{arch}.expert_shared_count` | u32 | COMPOSE | Shared experts (Qwen2MoE, DeepSeek2) |
| `{arch}.expert_shared_feed_forward_length` | u32 | COMPILE | Shared expert FFN size |
| `{arch}.expert_gating_func` | u32 | COMPOSE | Routing function type |
| `{arch}.expert_weights_scale` | f32 | COMPOSE | Expert weight scaling |
| `{arch}.expert_weights_norm` | bool | COMPOSE | Normalize expert weights |
| `{arch}.expert_group_count` | u32 | COMPOSE | Expert groups (DeepSeek3) |
| `{arch}.expert_group_used_count` | u32 | COMPOSE | Groups selected per token |
| `{arch}.expert_group_scale` | f32 | COMPOSE | Group scaling |
| `{arch}.experts_per_group` | u32 | COMPOSE | Experts per group |
| `{arch}.expert_chunk_feed_forward_length` | u32 | COMPILE | Chunked expert FFN |
| `{arch}.interleave_moe_layer_step` | u32 | COMPOSE | MoE layer interleaving pattern |
| `{arch}.moe_every_n_layers` | u32 | COMPOSE | MoE frequency |
| `{arch}.moe_latent_size` | u32 | COMPILE | MoE latent compression |
| `{arch}.router_logit_softcapping` | f32 | COMPOSE | Router softcap |
| `{arch}.use_parallel_residual` | bool | COMPOSE | Parallel attn+FFN (GPT-J/NeoX) |
| `{arch}.swiglu_clamp_exp` | f32 | COMPOSE | SwiGLU clamping |
| `{arch}.swiglu_clamp_shexp` | f32 | COMPOSE | SwiGLU shift-exp clamping |

## 7. SSM / Mamba Configuration

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `{arch}.ssm.conv_kernel` | u32 | COMPILE | Conv1d kernel size |
| `{arch}.ssm.inner_size` | u32 | COMPILE | SSM inner dimension |
| `{arch}.ssm.state_size` | u32 | COMPILE | State space dimension |
| `{arch}.ssm.time_step_rank` | u32 | COMPILE | dt projection rank |
| `{arch}.ssm.group_count` | u32 | COMPILE | Group count (Mamba2) |
| `{arch}.ssm.dt_b_c_rms` | bool | COMPOSE | RMS norm on dt/B/C |
| `{arch}.shortconv.l_cache` | u32 | COMPILE | Short conv cache length |

## 8. RWKV Configuration

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `{arch}.wkv.head_size` | u32 | COMPILE | WKV head dimension |
| `{arch}.time_mix_extra_dim` | u32 | COMPILE | LoRA rank for time mixing |
| `{arch}.time_decay_extra_dim` | u32 | COMPILE | LoRA rank for time decay |
| `{arch}.token_shift_count` | u32 | COMPOSE | Number of token shifts |

## 9. DeepSeek2 MLA Configuration

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `{arch}.attention.key_length_mla` | u32 | COMPILE | KV LoRA compressed dim |
| `{arch}.attention.value_length_mla` | u32 | COMPILE | Value LoRA dim |
| `{arch}.attention.kv_lora_rank` | u32 | COMPILE | KV LoRA rank |
| `{arch}.attention.q_lora_rank` | u32 | COMPILE | Q LoRA rank |
| `{arch}.leading_dense_block_count` | u32 | COMPOSE | Dense layers before MLA kicks in |
| `{arch}.attention.decay_lora_rank` | u32 | COMPILE | Decay LoRA (KDA) |
| `{arch}.attention.gate_lora_rank` | u32 | COMPILE | Gate LoRA (KDA) |
| `{arch}.attention.iclr_lora_rank` | u32 | COMPILE | ICLR LoRA |
| `{arch}.kda.head_dim` | u32 | COMPILE | KDA head dimension |

## 10. DeltaNet Configuration

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `{arch}.attention.head_count_deltanet` | u32 | COMPILE | DeltaNet head count |
| `{arch}.attention.key_length_deltanet` | u32 | COMPILE | DeltaNet key dim |
| `{arch}.attention.value_length_deltanet` | u32 | COMPILE | DeltaNet value dim |
| `{arch}.attention.conv_kernel_deltanet` | u32 | COMPILE | DeltaNet conv kernel size |

## 11. Encoder-Decoder (T5, BERT)

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `{arch}.decoder_block_count` | u32 | COMPOSE | Decoder layer count |
| `{arch}.decoder_start_token_id` | u32 | DISPATCH | First decoder token |
| `{arch}.attention.relative_buckets_count` | u32 | COMPILE | T5 relative position bias |
| `{arch}.pooling_type` | u32 | COMPOSE | Embedding pooling (BERT) |
| `enc.output_norm.weight` (tensor) | - | COMPOSE | Encoder output exists |

## 12. Vision / Audio / Multimodal

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `{arch}.audio.n_fft` | u32 | COMPILE | Audio FFT size |
| `{arch}.audio.hop_length` | u32 | COMPILE | Audio hop length |
| `{arch}.audio.n_mels` | u32 | COMPILE | Mel filterbank bins |
| `{arch}.convnext.block_count` | u32 | COMPOSE | ConvNext blocks (WavTokenizer) |
| `{arch}.convnext.embedding_length` | u32 | COMPILE | ConvNext hidden dim |
| `{arch}.posnet.block_count` | u32 | COMPOSE | PosNet blocks (WavTokenizer) |
| `{arch}.posnet.embedding_length` | u32 | COMPILE | PosNet hidden dim |
| `{arch}.features_length` | u32 | COMPILE | Feature extractor dim |
| `{arch}.dense_2_feat_in/out` | u32 | COMPILE | Dense layer dims |
| `{arch}.dense_3_feat_in/out` | u32 | COMPILE | Dense layer dims |

## 13. Special / Activation

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `{arch}.tensor_data_layout` | string | COMPILE | Weight layout (affects matvec kernel) |
| `{arch}.n_deepstack_layers` | u32 | COMPOSE | DeepStack layer structure |
| `{arch}.nextn_predict_layers` | u32 | COMPOSE | Next-N prediction layers |
| `{arch}.attention.value_residual_mix_lora_rank` | u32 | COMPOSE | Value residual mixing |
| `xielu.alpha_n/alpha_p/beta/eps` | f32 | COMPOSE | XiELU activation params |
| `{arch}.attention.indexer.*` | u32 | COMPOSE | Indexer attention (novel arch) |

## 14. Tokenizer (relevant for test/correctness, not kernel composition)

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `tokenizer.ggml.bos_token_id` | u32 | DISPATCH | First token for standalone decode |
| `tokenizer.ggml.eos_token_id` | u32 | DISPATCH | Stop token detection |
| `tokenizer.ggml.model` | string | - | Tokenizer type (BPE, SPM, etc.) |

## 15. Sampling Defaults (not kernel composition, but useful context)

| GGUF Key | Type | Informs | How |
|----------|------|---------|-----|
| `general.sampling.temp` | f32 | - | Default temperature |
| `general.sampling.top_k` | u32 | - | Default top-k |
| `general.sampling.top_p` | f32 | - | Default top-p |
| `general.sampling.min_p` | f32 | - | Default min-p |

## Summary: What We Read vs What We Don't

### Currently read by our model loader (43 keys):
Architecture, core dims, attention heads, RoPE params, norm epsilon,
MoE counts, SSM params, RWKV params, DeltaNet params, SWA window,
ALiBi bias, attention scale, softcaps, parallel residual, YaRN params,
audio params, T5 params, Chameleon params, embed scale

### NOT yet read but available (140+ keys):
context_length, key/value_length explicit, causal flag, activation function,
SWA pattern, clamp_kqv, group norm, shared KV layers, temperature attention,
full_attention_interval, all MoE expert params (groups, scaling, gating func,
interleaving), swin_norm, residual_scale, rescale_every_n, logit_scale,
tensor_data_layout, deepstack, nextn predict, value residual mixing,
decoder_start_token_id, pooling_type, vision/audio dims, XiELU params,
indexer attention, all sampling defaults, rope sections, rope SWA-specific dims
