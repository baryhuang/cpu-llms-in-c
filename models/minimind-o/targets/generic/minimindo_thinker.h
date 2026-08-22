#ifndef LLM_IN_C_MINIMINDO_THINKER_H
#define LLM_IN_C_MINIMINDO_THINKER_H

#include <stddef.h>
#include <stdint.h>

typedef struct minimindo_thinker minimindo_thinker;

minimindo_thinker *minimindo_thinker_open(const char *image_path,
                                          uint32_t max_context,
                                          char *error, size_t error_capacity);
void minimindo_thinker_close(minimindo_thinker *model);
void minimindo_thinker_reset(minimindo_thinker *model);

uint32_t minimindo_thinker_vocab_size(const minimindo_thinker *model);
uint32_t minimindo_thinker_position(const minimindo_thinker *model);
uint32_t minimindo_thinker_hidden_size(const minimindo_thinker *model);

int minimindo_thinker_forward(minimindo_thinker *model, uint32_t token_id,
                              float *logits, size_t logits_count,
                              char *error, size_t error_capacity);

/*
 * Native MiniMind-O bridge API.  bridge_states receives the hidden state after
 * the bridge layer selected by the official checkpoint (layer 3).  The
 * embedding entry point is used for audio-pad positions after the frozen
 * SenseVoice encoder and audio projector replace their token embeddings.
 */
int minimindo_thinker_forward_bridge(minimindo_thinker *model,
                                     uint32_t token_id,
                                     float *logits, size_t logits_count,
                                     float *bridge_states,
                                     size_t bridge_count,
                                     char *error, size_t error_capacity);

int minimindo_thinker_forward_embedding(minimindo_thinker *model,
                                        const float *embedding,
                                        size_t embedding_count,
                                        float *logits, size_t logits_count,
                                        float *bridge_states,
                                        size_t bridge_count,
                                        char *error, size_t error_capacity);

/* Prefill-only variants update all recurrent state but skip the unused LM head. */
int minimindo_thinker_prefill_bridge(minimindo_thinker *model,
                                     uint32_t token_id,
                                     float *bridge_states,
                                     size_t bridge_count,
                                     char *error, size_t error_capacity);
int minimindo_thinker_prefill_embedding(minimindo_thinker *model,
                                        const float *embedding,
                                        size_t embedding_count,
                                        float *bridge_states,
                                        size_t bridge_count,
                                        char *error, size_t error_capacity);

/*
 * Process a complete prompt layer-by-layer.  replacement_mask selects rows in
 * replacement_embeddings (token_count * hidden floats), which is how projected
 * audio embeddings replace audio-pad token embeddings.  Every bridge state is
 * returned in sequence order; when logits is non-NULL it contains only the
 * final position. Passing NULL skips the language head for deterministic
 * audio-drain precomputation.
 */
int minimindo_thinker_prefill_sequence(
    minimindo_thinker *model,
    const uint32_t *token_ids, size_t token_count,
    const float *replacement_embeddings, const uint8_t *replacement_mask,
    float *logits, size_t logits_count,
    float *bridge_states, size_t bridge_count,
    char *error, size_t error_capacity);

#endif
