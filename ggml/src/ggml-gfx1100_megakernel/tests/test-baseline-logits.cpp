// test-baseline-logits.cpp — Dump baseline llama_decode logits to binary file
// for comparison against megakernel output.
//
// Usage: test-baseline-logits <model.gguf> <output.bin> [n_tokens]
// Feeds BOS then 9 greedy tokens (10 total), saves all logits.
//
// Build WITHOUT GGML_GFX1100_MEGAKERNEL to avoid GPU conflict.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <output.bin> [n_tokens]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];
    int n_tokens = argc > 3 ? atoi(argv[3]) : 10;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 8192;  // match megakernel max_seq_len for fair SU-RoPE freq factor comparison
    cp.n_batch = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    FILE * fout = fopen(out_path, "wb");
    if (!fout) { fprintf(stderr, "Cannot open %s\n", out_path); return 1; }

    // Write header: n_tokens, n_vocab
    fwrite(&n_tokens, sizeof(int), 1, fout);
    fwrite(&n_vocab, sizeof(int), 1, fout);

    // Use model's actual BOS token (fallback to 1 for models without BOS)
    int cur_token = llama_vocab_bos(vocab);
    if (cur_token < 0 || cur_token >= n_vocab) cur_token = 1;
    fprintf(stderr, "Starting token: %d\n", cur_token);

    for (int pos = 0; pos < n_tokens; pos++) {
        llama_batch batch = llama_batch_get_one(&cur_token, 1);
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "llama_decode failed at pos=%d\n", pos);
            fclose(fout);
            return 1;
        }
        const float * logits = llama_get_logits(ctx);

        // Write logits for this position
        fwrite(logits, sizeof(float), n_vocab, fout);

        // Write token info
        int argmax = 0; float maxv = logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > maxv) { maxv = logits[i]; argmax = i; }
        }

        char buf[64] = {};
        llama_token_to_piece(vocab, argmax, buf, sizeof(buf) - 1, 0, true);
        fprintf(stderr, "  [pos=%d] token=%d → %d \"%s\" (%.2f)\n", pos, cur_token, argmax, buf, maxv);

        fprintf(stderr, "    logits[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                logits[0], logits[1], logits[2], logits[3],
                logits[4], logits[5], logits[6], logits[7]);

        // Also dump hidden state of intermediate tensors via eval callback
        // This requires accessing internal ggml graph which we can't do here.
        // Instead, dump the full logit stats
        float sum = 0, sumsq = 0;
        for (int i = 0; i < n_vocab; i++) { sum += logits[i]; sumsq += logits[i]*logits[i]; }
        float mean = sum / n_vocab, rms = sqrtf(sumsq / n_vocab);
        fprintf(stderr, "    logit_mean=%.4f logit_rms=%.4f top=%.4f\n", mean, rms, maxv);

        fwrite(&cur_token, sizeof(int), 1, fout);
        fwrite(&argmax, sizeof(int), 1, fout);

        cur_token = argmax;
    }

    fclose(fout);
    fprintf(stderr, "Saved %d positions to %s\n", n_tokens, out_path);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
