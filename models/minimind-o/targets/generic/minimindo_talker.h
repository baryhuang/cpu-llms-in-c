#ifndef LLM_IN_C_MINIMINDO_TALKER_H
#define LLM_IN_C_MINIMINDO_TALKER_H

#include <stddef.h>
#include <stdint.h>

enum { MINIMINDO_AUDIO_CODEBOOKS = 8 };

typedef struct minimindo_talker minimindo_talker;

minimindo_talker *minimindo_talker_open(const char *image_path,
                                        uint32_t max_context,
                                        char *error, size_t error_capacity);
void minimindo_talker_close(minimindo_talker *model);
void minimindo_talker_reset(minimindo_talker *model);

uint32_t minimindo_talker_hidden_size(const minimindo_talker *model);
uint32_t minimindo_talker_vocab_size(const minimindo_talker *model);
uint32_t minimindo_talker_pad_token(const minimindo_talker *model);
uint32_t minimindo_talker_position(const minimindo_talker *model);

/* logits is codebook-major: 8 consecutive audio-vocabulary vectors. */
int minimindo_talker_forward(minimindo_talker *model,
                             const float *bridge_states,
                             size_t bridge_count,
                             const uint32_t audio_ids[MINIMINDO_AUDIO_CODEBOOKS],
                             const float *speaker_embedding,
                             size_t speaker_count,
                             float *logits, size_t logits_count,
                             char *error, size_t error_capacity);

/* Only selected codebook heads are evaluated; mask zero is prefill-only. */
int minimindo_talker_forward_masked(
    minimindo_talker *model,
    const float *bridge_states,
    size_t bridge_count,
    const uint32_t audio_ids[MINIMINDO_AUDIO_CODEBOOKS],
    const float *speaker_embedding,
    size_t speaker_count,
    uint32_t logits_mask,
    float *logits, size_t logits_count,
    char *error, size_t error_capacity);

/* Prompt-only sequence path: one fixed audio-codebook tuple, no output heads. */
int minimindo_talker_prefill_sequence(
    minimindo_talker *model,
    const float *bridge_states, size_t bridge_count,
    const uint32_t audio_ids[MINIMINDO_AUDIO_CODEBOOKS],
    size_t token_count,
    char *error, size_t error_capacity);

#endif
