#include "minimax_h3_m3_aot.h"

#include <math.h>
#include <string.h>

static int checked_add_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (UINT64_MAX - left < right) {
        return 0;
    }
    *result = left + right;
    return 1;
}

static int checked_mul_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (left != 0u && right > UINT64_MAX / left) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static int align_u64(uint64_t value, uint64_t alignment, uint64_t *result) {
    uint64_t remainder = value % alignment;
    if (remainder == 0u) {
        *result = value;
        return 1;
    }
    return checked_add_u64(value, alignment - remainder, result);
}

static int equal_float_bits(float left, float right) {
    uint32_t left_bits;
    uint32_t right_bits;
    memcpy(&left_bits, &left, sizeof(left_bits));
    memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

minimax_h3_status minimax_h3_m3_turbo4_aot_plan_make(
    const float video_timesteps[MINIMAX_H3_M3_TURBO_EVALUATIONS],
    const float audio_timesteps[MINIMAX_H3_M3_TURBO_EVALUATIONS],
    minimax_h3_m3_aot_plan *plan) {
    static const uint8_t slot_modalities[MINIMAX_H3_MODALITY_COUNT] = {
        MINIMAX_H3_VIDEO_TAG, MINIMAX_H3_TEXT_TAG, MINIMAX_H3_AUDIO_TAG
    };
    minimax_h3_m3_aot_plan result;
    uint64_t slot_bytes;
    uint64_t block_bytes;
    uint64_t offset;
    size_t evaluation;
    size_t modality_index;
    size_t slot_index = 0u;
    size_t final_count = 0u;

    if (video_timesteps == NULL || audio_timesteps == NULL || plan == NULL) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    memset(&result, 0, sizeof(result));
    for (evaluation = 0u; evaluation < MINIMAX_H3_M3_TURBO_EVALUATIONS;
         ++evaluation) {
        if (!isfinite(video_timesteps[evaluation]) ||
            !isfinite(audio_timesteps[evaluation]) ||
            video_timesteps[evaluation] < 0.0f ||
            video_timesteps[evaluation] > 1.0f ||
            audio_timesteps[evaluation] < 0.0f ||
            audio_timesteps[evaluation] > 1.0f) {
            return MINIMAX_H3_INVALID_ARGUMENT;
        }
        if (evaluation > 0u &&
            (video_timesteps[evaluation] <= video_timesteps[evaluation - 1u] ||
             audio_timesteps[evaluation] <= audio_timesteps[evaluation - 1u])) {
            return MINIMAX_H3_INVALID_ARGUMENT;
        }
    }

    if (!checked_mul_u64(MINIMAX_H3_M3_ADALN_FIELDS,
                         MINIMAX_H3_M3_HIDDEN_SIZE, &slot_bytes) ||
        !checked_mul_u64(slot_bytes, MINIMAX_H3_M3_FP16_BYTES, &slot_bytes) ||
        !checked_mul_u64(slot_bytes, MINIMAX_H3_M3_ACTIVE_MODULATION_SLOTS,
                         &block_bytes) ||
        !checked_mul_u64(block_bytes, MINIMAX_H3_M3_BLOCK_COUNT,
                         &result.modulation_bytes)) {
        return MINIMAX_H3_OVERFLOW;
    }
    result.header_bytes = MINIMAX_H3_M3_AOT_ALIGNMENT;
    result.modulation_offset = result.header_bytes;

    for (evaluation = 0u; evaluation < MINIMAX_H3_M3_TURBO_EVALUATIONS;
         ++evaluation) {
        for (modality_index = 0u; modality_index < MINIMAX_H3_MODALITY_COUNT;
             ++modality_index) {
            uint8_t modality = slot_modalities[modality_index];
            minimax_h3_m3_modulation_slot *slot =
                &result.modulation_slots[slot_index];
            slot->evaluation = (uint8_t)evaluation;
            slot->modality = modality;
            slot->schedule_kind = modality == MINIMAX_H3_AUDIO_TAG
                                      ? MINIMAX_H3_M3_AUDIO_SCHEDULE
                                      : MINIMAX_H3_M3_VIDEO_SCHEDULE;
            slot->timestep = modality == MINIMAX_H3_AUDIO_TAG
                                 ? audio_timesteps[evaluation]
                                 : video_timesteps[evaluation];
            slot->byte_offset = (uint64_t)slot_index * slot_bytes;
            result.evaluation_modulation_slot[evaluation][modality] =
                (uint8_t)slot_index;
            ++slot_index;
        }
    }

    if (!checked_add_u64(result.modulation_offset, result.modulation_bytes,
                         &offset) ||
        !align_u64(offset, MINIMAX_H3_M3_AOT_ALIGNMENT,
                   &result.final_norm_offset)) {
        return MINIMAX_H3_OVERFLOW;
    }

    for (evaluation = 0u; evaluation < MINIMAX_H3_M3_TURBO_EVALUATIONS;
         ++evaluation) {
        float candidates[2] = {
            video_timesteps[evaluation], audio_timesteps[evaluation]
        };
        size_t output_kind;
        for (output_kind = 0u; output_kind < 2u; ++output_kind) {
            size_t final_index;
            for (final_index = 0u; final_index < final_count; ++final_index) {
                if (equal_float_bits(result.final_slots[final_index].timestep,
                                     candidates[output_kind])) {
                    break;
                }
            }
            if (final_index == final_count) {
                uint64_t final_slot_bytes;
                if (final_count >= MINIMAX_H3_M3_FINAL_TIMESTEP_SLOTS ||
                    !checked_mul_u64(MINIMAX_H3_M3_FINAL_NORM_FIELDS,
                                     MINIMAX_H3_M3_HIDDEN_SIZE,
                                     &final_slot_bytes) ||
                    !checked_mul_u64(final_slot_bytes,
                                     MINIMAX_H3_M3_FP16_BYTES,
                                     &final_slot_bytes)) {
                    return MINIMAX_H3_OVERFLOW;
                }
                result.final_slots[final_index].timestep = candidates[output_kind];
                result.final_slots[final_index].byte_offset =
                    (uint64_t)final_index * final_slot_bytes;
                ++final_count;
            }
            result.evaluation_final_slot[evaluation][output_kind] =
                (uint8_t)final_index;
        }
    }
    if (final_count != MINIMAX_H3_M3_FINAL_TIMESTEP_SLOTS ||
        !checked_mul_u64(MINIMAX_H3_M3_FINAL_TIMESTEP_SLOTS,
                         MINIMAX_H3_M3_FINAL_NORM_FIELDS,
                         &result.final_norm_bytes) ||
        !checked_mul_u64(result.final_norm_bytes,
                         MINIMAX_H3_M3_HIDDEN_SIZE,
                         &result.final_norm_bytes) ||
        !checked_mul_u64(result.final_norm_bytes, MINIMAX_H3_M3_FP16_BYTES,
                         &result.final_norm_bytes) ||
        !checked_add_u64(result.final_norm_offset, result.final_norm_bytes,
                         &offset) ||
        !align_u64(offset, MINIMAX_H3_M3_AOT_ALIGNMENT, &result.file_bytes)) {
        return MINIMAX_H3_OVERFLOW;
    }

    *plan = result;
    return MINIMAX_H3_OK;
}
