// arch_ids.h — gfx1100 megakernel architecture + capability ID macros.
//
// These IDs are used for compile-time specialization of the megakernel:
// each (arch, capability) tuple hashes to its own .hsaco, so hipcc can DCE
// untaken branches. The IDs mirror baseline's enum llm_arch in
// src/llama-arch.h — keep them in the same order (new archs appended at end).
//
// Host code (gfx1100-megakernel.cpp) maps LLM_ARCH_* → ARCH_* here.
// Device code (decode.hip / prefill.hip) does `#if ARCH_ID == ARCH_LLAMA`.

#pragma once

// ----------------------------------------------------------------------------
// Architecture IDs — mirror src/llama-arch.h::llm_arch
// ----------------------------------------------------------------------------
#define ARCH_UNKNOWN         0
#define ARCH_CLIP            1
#define ARCH_LLAMA           2
#define ARCH_LLAMA4          3
#define ARCH_DECI            4
#define ARCH_FALCON          5
#define ARCH_BAICHUAN        6
#define ARCH_GROK            7
#define ARCH_GPT2            8
#define ARCH_GPTJ            9
#define ARCH_GPTNEOX         10
#define ARCH_MPT             11
#define ARCH_STARCODER       12
#define ARCH_REFACT          13
#define ARCH_BERT            14
#define ARCH_MODERN_BERT     15
#define ARCH_NOMIC_BERT      16
#define ARCH_NOMIC_BERT_MOE  17
#define ARCH_NEO_BERT        18
#define ARCH_JINA_BERT_V2    19
#define ARCH_JINA_BERT_V3    20
#define ARCH_EUROBERT        21
#define ARCH_BLOOM           22
#define ARCH_STABLELM        23
#define ARCH_QWEN            24
#define ARCH_QWEN2           25
#define ARCH_QWEN2MOE        26
#define ARCH_QWEN2VL         27
#define ARCH_QWEN3           28
#define ARCH_QWEN3MOE        29
#define ARCH_QWEN3NEXT       30
#define ARCH_QWEN3VL         31
#define ARCH_QWEN3VLMOE      32
#define ARCH_QWEN35          33
#define ARCH_QWEN35MOE       34
#define ARCH_PHI2            35
#define ARCH_PHI3            36
#define ARCH_PHIMOE          37
#define ARCH_PLAMO           38
#define ARCH_PLAMO2          39
#define ARCH_PLAMO3          40
#define ARCH_CODESHELL       41
#define ARCH_ORION           42
#define ARCH_INTERNLM2       43
#define ARCH_MINICPM         44
#define ARCH_MINICPM3        45
#define ARCH_GEMMA           46
#define ARCH_GEMMA2          47
#define ARCH_GEMMA3          48
#define ARCH_GEMMA3N         49
#define ARCH_GEMMA4          50
#define ARCH_GEMMA_EMBEDDING 51
#define ARCH_STARCODER2      52
#define ARCH_MAMBA           53
#define ARCH_MAMBA2          54
#define ARCH_JAMBA           55
#define ARCH_FALCON_H1       56
#define ARCH_XVERSE          57
#define ARCH_COMMAND_R       58
#define ARCH_COHERE2         59
#define ARCH_DBRX            60
#define ARCH_OLMO            61
#define ARCH_OLMO2           62
#define ARCH_OLMOE           63
#define ARCH_OPENELM         64
#define ARCH_ARCTIC          65
#define ARCH_DEEPSEEK        66
#define ARCH_DEEPSEEK2       67
#define ARCH_DEEPSEEK2OCR    68
#define ARCH_CHATGLM         69
#define ARCH_GLM4            70
#define ARCH_GLM4_MOE        71
#define ARCH_GLM_DSA         72
#define ARCH_BITNET          73
#define ARCH_T5              74
#define ARCH_T5ENCODER       75
#define ARCH_JAIS            76
#define ARCH_JAIS2           77
#define ARCH_NEMOTRON        78
#define ARCH_NEMOTRON_H      79
#define ARCH_NEMOTRON_H_MOE  80
#define ARCH_EXAONE          81
#define ARCH_EXAONE4         82
#define ARCH_EXAONE_MOE      83
#define ARCH_RWKV6           84
#define ARCH_RWKV6QWEN2      85
#define ARCH_RWKV7           86
#define ARCH_ARWKV7          87
#define ARCH_GRANITE         88
#define ARCH_GRANITE_MOE     89
#define ARCH_GRANITE_HYBRID  90
#define ARCH_CHAMELEON       91
#define ARCH_WAVTOKENIZER_DEC 92
#define ARCH_PLM             93
#define ARCH_BAILINGMOE      94
#define ARCH_BAILINGMOE2     95
#define ARCH_DOTS1           96
#define ARCH_ARCEE           97
#define ARCH_AFMOE           98
#define ARCH_ERNIE4_5        99
#define ARCH_ERNIE4_5_MOE    100
#define ARCH_HUNYUAN_MOE     101
#define ARCH_HUNYUAN_DENSE   102
#define ARCH_SMOLLM3         103
#define ARCH_OPENAI_MOE      104
#define ARCH_LFM2            105
#define ARCH_LFM2MOE         106
#define ARCH_DREAM           107
#define ARCH_SMALLTHINKER    108
#define ARCH_LLADA           109
#define ARCH_LLADA_MOE       110
#define ARCH_SEED_OSS        111
#define ARCH_GROVEMOE        112
#define ARCH_APERTUS         113
#define ARCH_MINIMAX_M2      114
#define ARCH_COGVLM          115
#define ARCH_RND1            116
#define ARCH_PANGU_EMBED     117
#define ARCH_MISTRAL3        118
#define ARCH_MISTRAL4        119
#define ARCH_PADDLEOCR       120
#define ARCH_MIMO2           121
#define ARCH_STEP35          122
#define ARCH_LLAMA_EMBED     123
#define ARCH_MAINCODER       124
#define ARCH_KIMI_LINEAR     125

// ----------------------------------------------------------------------------
// RoPE types — mirror include/llama.h::llama_rope_type
//   LLAMA_ROPE_TYPE_NONE   = -1
//   LLAMA_ROPE_TYPE_NORM   = 0
//   LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX  (2)
//   LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE (8)
//   LLAMA_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE (24)
//   LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION (24 | 8 = ...)
// We use our own dense encoding for macro-readable #if.
// ----------------------------------------------------------------------------
#define ROPE_NONE    0
#define ROPE_NORM    1
#define ROPE_NEOX    2
#define ROPE_MROPE   3
#define ROPE_IMROPE  4
#define ROPE_VISION  5

// ----------------------------------------------------------------------------
// Norm types
// ----------------------------------------------------------------------------
#define NORM_RMS   0
#define NORM_L2    1
#define NORM_LAYER 2
#define NORM_GROUP 3

// ----------------------------------------------------------------------------
// Activation types — mirror ggml_unary_op where relevant
// ----------------------------------------------------------------------------
#define ACT_NONE        0
#define ACT_SILU        1  // Llama family, Qwen, Mistral
#define ACT_GELU        2  // exact GELU
#define ACT_GELU_TANH   3  // approximate (Gemma, some others)
#define ACT_GELU_QUICK  4
#define ACT_RELU        5  // BLOOM, RefACT
#define ACT_RELU2       6  // squared ReLU (some Phi variants)
#define ACT_SWISH       ACT_SILU

// ----------------------------------------------------------------------------
// Attention scale types
// ----------------------------------------------------------------------------
#define ATTN_SCALE_DEFAULT 0  // 1 / sqrt(head_dim)
#define ATTN_SCALE_SOFTCAP 1  // Gemma2: tanh(logits / cap) * cap
#define ATTN_SCALE_CUSTOM  2  // hparams.f_attention_scale override

// ----------------------------------------------------------------------------
// SWA types — mirror src/llama-hparams.h::llama_swa_type
// ----------------------------------------------------------------------------
#define SWA_NONE      0
#define SWA_STANDARD  1
#define SWA_CHUNKED   2
#define SWA_SYMMETRIC 3

// ----------------------------------------------------------------------------
// Per-layer types (mirrors cfg.layer_types[])
// ----------------------------------------------------------------------------
#define LAYER_ATTENTION 0
#define LAYER_DELTANET  1
#define LAYER_SSM       2
#define LAYER_RWKV      3
