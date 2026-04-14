// test-baseline-dump.cpp — Dump ALL baseline layer-0 intermediates
//
// Usage: test-baseline-dump <model.gguf> [output_dir]
//
// Runs BOS token through standard llama.cpp baseline, dumps every tensor
// that touches layer 0 to binary files. Each file is raw f32 array.
//
// Output files: {output_dir}/bl_{counter:03d}_{name}.bin
// Also dumps src[0] of each tensor to capture matvec inputs that aren't
// separately named in the graph (e.g., O-proj output before norm).
//
// Build with rocWMMA, same flags as megakernel. Intended to be built ONCE
// and reused across sessions without recompilation.
#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static const char * g_out_dir = ".";
static int g_dump_position = 0;   // only dump at this position
static int g_current_position = -1;  // tracks current decode position
static bool g_dump_active = false;

static void dump_tensor(const char * tag, const ggml_tensor * t) {
    int64_t n = ggml_nelements(t);
    if (n > 1000000) return;  // skip huge tensors (vocab logits etc for non-output)
    std::vector<float> data(n);

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h16(n);
        ggml_backend_tensor_get(t, h16.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++) data[i] = ggml_fp16_to_fp32(h16[i]);
    } else {
        return;  // skip non-float types
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/bl_%s.bin", g_out_dir, tag);
    FILE * f = fopen(path, "wb");
    if (f) { fwrite(data.data(), sizeof(float), n, f); fclose(f); }

    double sum = 0, sumsq = 0;
    for (int64_t i = 0; i < n; i++) { sum += data[i]; sumsq += data[i]*data[i]; }
    fprintf(stderr, "  %-40s [%6lld] first4=[%12.6f %12.6f %12.6f %12.6f] mean=%10.6f rms=%10.6f\n",
            tag, (long long)n,
            n > 0 ? data[0] : 0, n > 1 ? data[1] : 0, n > 2 ? data[2] : 0, n > 3 ? data[3] : 0,
            sum/n, sqrt(sumsq/n));
}

static bool eval_cb(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (!g_dump_active) return ask ? true : true;  // compute but don't dump
    if (ask) {
        const char * name = t->name;
        if (name && strstr(name, "norm") && strstr(name, "-0") && t->src[0]) {
            static int pre_counter = 0;
            char tag[256];
            snprintf(tag, sizeof(tag), "PRE_%03d_%s_input", pre_counter++, name);
            for (char * p = tag; *p; p++) {
                if (*p == '-' || *p == ' ') *p = '_';
                if (*p == '(' || *p == ')') *p = '_';
            }
            // src[0] has been computed by this point (ask phase runs after src computation)
            dump_tensor(tag, t->src[0]);
        }
        return true;
    }

    const char * name = t->name;
    if (!name || !name[0]) return true;

    // Dump all layers and global tensors (for divergence bisection)
    // Filter: only dump attention-related and layer-end tensors to keep output manageable
    bool is_attn = (strstr(name, "attn_v") != nullptr || strstr(name, "kqv") != nullptr ||
                    strstr(name, "attn_out") != nullptr || strstr(name, "ffn_out") != nullptr ||
                    strstr(name, "l_out") != nullptr);
    bool is_global = (strstr(name, "inp_embd") != nullptr ||
                      strstr(name, "result_norm") != nullptr ||
                      strstr(name, "result_output") != nullptr);
    // Always dump layer 0/1 (detailed), and attn/ffn outputs for all layers
    bool is_layer0 = (strstr(name, "-0") != nullptr);
    bool is_layer1 = (strstr(name, "-1") != nullptr && strstr(name, "-1 ") == nullptr
                      && strstr(name, "-10") == nullptr && strstr(name, "-11") == nullptr
                      && strstr(name, "-12") == nullptr && strstr(name, "-13") == nullptr
                      && strstr(name, "-14") == nullptr && strstr(name, "-15") == nullptr
                      && strstr(name, "-16") == nullptr && strstr(name, "-17") == nullptr
                      && strstr(name, "-18") == nullptr && strstr(name, "-19") == nullptr);
    if (!is_layer0 && !is_layer1 && !is_global && !is_attn) return true;

    // Counter for unique filenames
    static int counter = 0;
    char tag[256];
    snprintf(tag, sizeof(tag), "%03d_%s", counter++, name);
    // Sanitize: replace - with _, space with _, parens with nothing
    for (char * p = tag; *p; p++) {
        if (*p == '-' || *p == ' ') *p = '_';
        if (*p == '(' || *p == ')') *p = '_';
    }

    dump_tensor(tag, t);

    // Also dump src[0] if it exists and is a different tensor — captures
    // intermediate values that aren't separately named (e.g., O-proj output
    // which is src[0] of the rmsnorm node)
    if (t->src[0] && t->src[0] != t && t->src[0]->name[0]) {
        char src_tag[256];
        snprintf(src_tag, sizeof(src_tag), "%03d_%s__src0_%s", counter-1, name, t->src[0]->name);
        for (char * p = src_tag; *p; p++) {
            if (*p == '-' || *p == ' ') *p = '_';
            if (*p == '(' || *p == ')') *p = '_';
        }
        // Always dump src[0] of layer-0 tensors — captures O-proj output etc.
        dump_tensor(src_tag, t->src[0]);
    }

    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [output_dir] [dump_position]\n", argv[0]);
        return 1;
    }
    if (argc >= 3) g_out_dir = argv[2];
    if (argc >= 4) g_dump_position = atoi(argv[3]);

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = 1;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    int cur_token = llama_vocab_bos(vocab);
    if (cur_token < 0) cur_token = 1;

    // If a baseline logit file is provided (argv[4]), read token sequence from it
    // instead of greedy decode. This ensures we feed the SAME tokens as the correctness test.
    const char * logit_file = argc >= 5 ? argv[4] : nullptr;
    std::vector<int> token_sequence;

    if (logit_file) {
        FILE * fin = fopen(logit_file, "rb");
        if (fin) {
            int bl_n_tokens, bl_n_vocab;
            fread(&bl_n_tokens, sizeof(int), 1, fin);
            fread(&bl_n_vocab, sizeof(int), 1, fin);
            token_sequence.resize(bl_n_tokens);
            for (int i = 0; i < bl_n_tokens; i++) {
                // Skip logits, read token_in and argmax
                fseek(fin, bl_n_vocab * sizeof(float), SEEK_CUR);
                int token_in, argmax;
                fread(&token_in, sizeof(int), 1, fin);
                fread(&argmax, sizeof(int), 1, fin);
                token_sequence[i] = token_in;
            }
            fclose(fin);
            fprintf(stderr, "Loaded %d tokens from %s\n", bl_n_tokens, logit_file);
        }
    }

    fprintf(stderr, "Running %s to position %d, dumping at that position...\n",
            token_sequence.empty() ? "greedy decode" : "token sequence from file", g_dump_position);

    for (int pos = 0; pos <= g_dump_position; pos++) {
        g_dump_active = (pos == g_dump_position);
        g_current_position = pos;

        // Use token from file if available, otherwise greedy decode
        if (!token_sequence.empty() && pos < (int)token_sequence.size()) {
            cur_token = token_sequence[pos];
        }

        llama_batch batch = llama_batch_get_one(&cur_token, 1);
        int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "llama_decode failed at pos %d: %d\n", pos, rc);
            return 1;
        }

        const float * logits = llama_get_logits(ctx);
        int argmax = 0;
        float maxv = logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > maxv) { maxv = logits[i]; argmax = i; }
        }

        {
            char buf[64] = {};
            llama_token_to_piece(vocab, argmax, buf, sizeof(buf)-1, 0, true);
            fprintf(stderr, "  [%d] in=%d argmax=%d \"%s\" (%.2f)%s\n", pos, cur_token, argmax, buf, maxv,
                    pos == g_dump_position ? " <-- DUMP" : "");
        }

        if (token_sequence.empty()) {
            cur_token = argmax;  // greedy decode
        }
    }

    fprintf(stderr, "Done. Dumps in %s/bl_*.bin\n", g_out_dir);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
