#include "minimindo_parallel.h"
#include "minimindo_talker.h"
#include "minimindo_thinker.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t argmax(const float *values, uint32_t count)
{
    uint32_t best = 0;
    for (uint32_t index = 1; index < count; ++index)
        if (values[index] > values[best]) best = index;
    return best;
}

static void usage(const char *program)
{
    fprintf(stderr, "usage: %s THINKER.mmo TALKER.mmo TOKEN [TOKEN ...]\n", program);
}

int main(int argc, char **argv)
{
    if (argc < 4) { usage(argv[0]); return 2; }
    char error[256] = {0};
    const uint32_t max_context = (uint32_t)(argc - 3 + 16);
    minimindo_thinker *thinker = minimindo_thinker_open(argv[1], max_context,
                                                         error, sizeof(error));
    if (thinker == NULL) { fprintf(stderr, "%s\n", error); return 3; }
    minimindo_talker *talker = minimindo_talker_open(argv[2], max_context,
                                                      error, sizeof(error));
    if (talker == NULL) { fprintf(stderr, "%s\n", error); return 3; }
    const uint32_t text_vocab = minimindo_thinker_vocab_size(thinker);
    const uint32_t audio_vocab = minimindo_talker_vocab_size(talker);
    const uint32_t hidden = minimindo_thinker_hidden_size(thinker);
    float *text_logits = malloc((size_t)text_vocab * sizeof(float));
    float *audio_logits = malloc((size_t)MINIMINDO_AUDIO_CODEBOOKS *
                                 audio_vocab * sizeof(float));
    float *bridge = malloc((size_t)hidden * sizeof(float));
    uint32_t audio_ids[MINIMINDO_AUDIO_CODEBOOKS];
    for (uint32_t i = 0; i < MINIMINDO_AUDIO_CODEBOOKS; ++i)
        audio_ids[i] = minimindo_talker_pad_token(talker);
    if (text_logits == NULL || audio_logits == NULL || bridge == NULL) return 3;
    if (minimindo_parallel_session_begin(4U) != 0) return 3;
    for (int index = 3; index < argc; ++index) {
        char *end = NULL;
        const unsigned long parsed = strtoul(argv[index], &end, 10);
        if (end == argv[index] || *end != '\0' || parsed >= text_vocab) {
            fprintf(stderr, "invalid token: %s\n", argv[index]);
            minimindo_parallel_session_end();
            return 2;
        }
        if (minimindo_thinker_forward_bridge(thinker, (uint32_t)parsed,
                text_logits, text_vocab, bridge, hidden, error, sizeof(error)) != 0 ||
            minimindo_talker_forward(talker, bridge, hidden, audio_ids, NULL, 0,
                audio_logits, (size_t)MINIMINDO_AUDIO_CODEBOOKS * audio_vocab,
                error, sizeof(error)) != 0) {
            fprintf(stderr, "%s\n", error);
            minimindo_parallel_session_end();
            return 4;
        }
    }
    printf("{\"text\":%u,\"audio\":[", argmax(text_logits, text_vocab));
    for (uint32_t i = 0; i < MINIMINDO_AUDIO_CODEBOOKS; ++i) {
        if (i != 0) putchar(',');
        printf("%u", argmax(audio_logits + (size_t)i * audio_vocab, audio_vocab));
    }
    printf("],\"positions\":%u}\n", minimindo_talker_position(talker));
    minimindo_parallel_session_end();
    free(text_logits); free(audio_logits); free(bridge);
    minimindo_talker_close(talker); minimindo_thinker_close(thinker);
    return 0;
}
