#ifndef MINIMAX_H3_M3_AOT_H
#define MINIMAX_H3_M3_AOT_H

#include <stddef.h>
#include <stdint.h>

#include "minimax_h3.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MINIMAX_H3_M3_BLOCK_COUNT = 50,
    MINIMAX_H3_M3_HIDDEN_SIZE = 5376,
    MINIMAX_H3_M3_ADALN_FIELDS = 6,
    MINIMAX_H3_M3_FINAL_NORM_FIELDS = 2,
    MINIMAX_H3_M3_TURBO_EVALUATIONS = 4,
    MINIMAX_H3_M3_ACTIVE_MODULATION_SLOTS = 12,
    MINIMAX_H3_M3_FINAL_TIMESTEP_SLOTS = 7,
    MINIMAX_H3_M3_AOT_ALIGNMENT = 16384,
    MINIMAX_H3_M3_FP16_BYTES = 2
};

typedef enum {
    MINIMAX_H3_M3_VIDEO_SCHEDULE = 0,
    MINIMAX_H3_M3_AUDIO_SCHEDULE = 1
} minimax_h3_m3_schedule_kind;

typedef struct {
    uint8_t evaluation;
    uint8_t modality;
    uint8_t schedule_kind;
    uint8_t reserved;
    float timestep;
    uint64_t byte_offset;
} minimax_h3_m3_modulation_slot;

typedef struct {
    float timestep;
    uint64_t byte_offset;
} minimax_h3_m3_final_slot;

typedef struct {
    uint64_t header_bytes;
    uint64_t modulation_offset;
    uint64_t modulation_bytes;
    uint64_t final_norm_offset;
    uint64_t final_norm_bytes;
    uint64_t file_bytes;
    minimax_h3_m3_modulation_slot
        modulation_slots[MINIMAX_H3_M3_ACTIVE_MODULATION_SLOTS];
    minimax_h3_m3_final_slot final_slots[MINIMAX_H3_M3_FINAL_TIMESTEP_SLOTS];
    uint8_t evaluation_modulation_slot[MINIMAX_H3_M3_TURBO_EVALUATIONS]
                                      [MINIMAX_H3_MODALITY_COUNT];
    uint8_t evaluation_final_slot[MINIMAX_H3_M3_TURBO_EVALUATIONS][2];
} minimax_h3_m3_aot_plan;

/*
 * Plan the FP16 tables consumed by the fixed Turbo-4 transformer graph.
 * Block table order is:
 *   [block][active slot][shift_msa, scale_msa, gate_msa,
 *                        shift_mlp, scale_mlp, gate_mlp][hidden].
 *
 * Only combinations reached by T2VA are emitted: video and text use the video
 * timestep; audio uses the audio timestep. The final norm table is deduplicated
 * by exact float32 timestep and stores [shift, scale][hidden].
 */
minimax_h3_status minimax_h3_m3_turbo4_aot_plan_make(
    const float video_timesteps[MINIMAX_H3_M3_TURBO_EVALUATIONS],
    const float audio_timesteps[MINIMAX_H3_M3_TURBO_EVALUATIONS],
    minimax_h3_m3_aot_plan *plan);

#ifdef __cplusplus
}
#endif

#endif
