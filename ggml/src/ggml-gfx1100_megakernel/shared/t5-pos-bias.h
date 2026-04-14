// shared/t5-pos-bias.h — T5 relative position bucket computation
// Ported from baseline src/llama-graph.cpp::llama_relative_position_bucket()
#pragma once
#include <cmath>

// Compute relative position bucket index for T5 relative position bias.
// bidirectional=true for encoder, false for decoder.
// Baseline: src/llama-graph.cpp lines 100-130
static inline int relative_position_bucket(int relative_position, int num_buckets,
                                            bool bidirectional, int max_distance = 128) {
    int ret = 0;
    int n = -relative_position;
    if (bidirectional) {
        num_buckets /= 2;
        ret += (n < 0) ? num_buckets : 0;
        n = abs(n);
    } else {
        n = (n < 0) ? 0 : n;
    }
    int max_exact = num_buckets / 2;
    if (n < max_exact) {
        ret += n;
    } else {
        ret += max_exact + (int)(logf((float)n / max_exact) / logf((float)max_distance / max_exact) *
                                  (num_buckets - max_exact));
        if (ret > num_buckets - 1) ret = num_buckets - 1;
    }
    return ret;
}
