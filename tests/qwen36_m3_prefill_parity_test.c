/* Live parity test between the batched prefill graph and the one-token
 * decode graph. Fast-math kernel specialization does not guarantee bitwise
 * equality across different graph shapes, so the gate matches the
 * repository verification standard: no NaN, bounded state drift, and an
 * identical next-token decision from the probe logits. Bitwise equality is
 * additionally reported as information. Requires the packed model. */

#include "qwen36_m3_decode.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CAPACITY = 64 };

/* Measured S32 fast-math reassociation drift reaches 0.084 absolute on
 * state values of magnitude ~20 while every next-token decision stays
 * identical; real defects (races, wrong indexing) produce NaNs or O(1)
 * errors. The S16 bucket is bitwise-exact and is asserted as such. */
static const double kStateTolerance = 0.25;

static const uint32_t kTokens[36] = {
    248045, 846, 198, 7734, 264, 351, 709, 514, 1866, 17, 1494, 264, 11,
    514, 292, 8, 421, 4523, 279, 7875, 869, 13, 8984, 1132, 279, 709, 13,
    248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271
};

static unsigned failures;

static size_t state_bytes(uint32_t kind) {
    switch (kind) {
    case QWEN36_M3_STATE_RECURRENT: return (size_t)48 * 128 * 128 * 4;
    case QWEN36_M3_STATE_CONVOLUTION: return (size_t)10240 * 4 * 4;
    default: return (size_t)CAPACITY * 4 * 256 * 4;
    }
}

static float **capture_states(qwen36_m3_model *model) {
    float **states = calloc(64 * 4, sizeof(float *));
    for (uint32_t layer = 0; layer < 64; ++layer) {
        for (uint32_t kind = 0; kind < 4; ++kind) {
            size_t bytes = state_bytes(kind);
            float *buffer = malloc(bytes);
            if (qwen36_m3_model_copy_state(model, layer, kind, buffer,
                                           bytes) == 0) {
                free(buffer);
                buffer = NULL;
            }
            states[layer * 4 + kind] = buffer;
        }
    }
    return states;
}

static uint32_t argmax(const float *values, size_t count) {
    uint32_t best = 0;
    for (uint32_t index = 1; index < count; ++index)
        if (values[index] > values[best]) best = index;
    return best;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <model-directory> <metallib>\n", argv[0]);
        return 2;
    }
    char error[512];
    qwen36_m3_model *model = qwen36_m3_model_open(
        argv[1], argv[2], CAPACITY, error, sizeof(error));
    if (model == NULL) {
        fprintf(stderr, "model open failed: %s\n", error);
        return 2;
    }
    qwen36_m3_decode_result result;
    const float *logits = NULL;
    size_t logit_count = 0;

    /* Token counts covering the S32 bucket, the S16 bucket, the single-token
     * tail, and the mixed 32+16 case within capacity 64. */
    const uint32_t runs[] = {16, 19, 32, 35, 48};
    for (size_t run = 0; run < sizeof(runs) / sizeof(runs[0]); ++run) {
        uint32_t count = runs[run];

        /* Reference: one-token forwards for tokens 0..count-1, then one
         * more forward for token count as the logits probe. */
        qwen36_m3_model_reset(model);
        for (uint32_t index = 0; index <= count; ++index) {
            if (qwen36_m3_model_forward(
                    model, kTokens[index % 36], index, &result, &logits,
                    &logit_count, error, sizeof(error)) != 0) {
                fprintf(stderr, "reference forward failed: %s\n", error);
                return 2;
            }
        }
        float **reference = capture_states(model);
        float *reference_logits = malloc(logit_count * sizeof(float));
        memcpy(reference_logits, logits, logit_count * sizeof(float));

        /* Candidates: the decode-identical exact path (QWEN36_PREFILL_MMA=0,
         * S16-only runs must stay bitwise), the float tiled simdgroup-matrix
         * path, and the half tiled path (argmax and tolerance gates only). */
        uint32_t sequence[64];
        for (uint32_t index = 0; index < count; ++index)
            sequence[index] = kTokens[index % 36];
        static const char *mode_names[3] = {"exact", "mma", "mma2"};
        static const char *mode_env[3] = {"0", "1", "2"};
        for (unsigned mode = 0; mode < 3; ++mode) {
            setenv("QWEN36_PREFILL_MMA", mode_env[mode], 1);
            qwen36_m3_model_reset(model);
            qwen36_m3_prefill_result prefill;
            if (qwen36_m3_model_prefill(model, sequence, count, 0,
                                        &prefill, error,
                                        sizeof(error)) != 0) {
                fprintf(stderr, "prefill failed: %s\n", error);
                return 2;
            }
            if (qwen36_m3_model_forward(
                    model, kTokens[count % 36], count, &result, &logits,
                    &logit_count, error, sizeof(error)) != 0) {
                fprintf(stderr, "candidate forward failed: %s\n", error);
                return 2;
            }

            double max_abs = 0.0;
            unsigned nan_count = 0;
            unsigned existence_mismatch = 0;
            int bitwise = 1;
            for (uint32_t layer = 0; layer < 64; ++layer) {
                for (uint32_t kind = 0; kind < 4; ++kind) {
                    size_t bytes = state_bytes(kind);
                    float *now = malloc(bytes);
                    size_t copied = qwen36_m3_model_copy_state(
                        model, layer, kind, now, bytes);
                    float *ref = reference[layer * 4 + kind];
                    if ((copied == 0) != (ref == NULL)) {
                        ++existence_mismatch;
                        free(now);
                        continue;
                    }
                    if (copied == 0) {
                        free(now);
                        continue;
                    }
                    if (memcmp(now, ref, copied) != 0) bitwise = 0;
                    size_t counts = copied / sizeof(float);
                    for (size_t index = 0; index < counts; ++index) {
                        if (isnan(now[index])) {
                            ++nan_count;
                            continue;
                        }
                        double difference = fabs((double)now[index] -
                                                 (double)ref[index]);
                        if (difference > max_abs) max_abs = difference;
                    }
                    free(now);
                }
            }
            uint32_t reference_best = argmax(reference_logits, logit_count);
            uint32_t candidate_best = argmax(logits, logit_count);
            int logits_bitwise = memcmp(reference_logits, logits,
                                        logit_count * sizeof(float)) == 0;
            int pass = nan_count == 0 && existence_mismatch == 0 &&
                       max_abs <= kStateTolerance &&
                       reference_best == candidate_best &&
                       (mode >= 1 || prefill.chunk32_count != 0 ||
                        bitwise);
            printf("check=run%u mode=%s chunks=32x%u/16x%u/1x%u "
                   "bitwise_states=%s bitwise_logits=%s "
                   "state_max_abs=%.9g nan=%u argmax=%u/%u status=%s\n",
                   count, mode_names[mode],
                   prefill.chunk32_count, prefill.chunk16_count,
                   prefill.single_count, bitwise ? "yes" : "no",
                   logits_bitwise ? "yes" : "no", max_abs, nan_count,
                   reference_best, candidate_best,
                   pass ? "PASS" : "FAIL");
            if (!pass) ++failures;
        }
        for (unsigned index = 0; index < 64 * 4; ++index)
            free(reference[index]);
        free(reference);
        free(reference_logits);
    }
    qwen36_m3_model_close(model);
    printf("VERDICT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
