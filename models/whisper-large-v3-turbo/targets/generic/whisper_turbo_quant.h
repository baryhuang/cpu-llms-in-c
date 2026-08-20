#ifndef CLLM_WHISPER_TURBO_QUANT_H
#define CLLM_WHISPER_TURBO_QUANT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Grouped weight records shared by the turbo encoder and decoder.
 *
 * Every quantized matrix row is a sequence of 128-column groups. Each group
 * record starts with a BF16 scale (2 bytes, little-endian):
 *
 *   Q4: 2 + 64  = 66 bytes  - signed 4-bit pairs, low nibble first
 *   Q8: 2 + 128 = 130 bytes - signed 8-bit values
 *   Q5: 2 + 64 + 16 = 82 bytes - low nibbles as Q4, then a 128-bit plane
 *       holding each element's 5th (sign) bit; value = 5-bit two's complement
 */
#define CLLM_WHISPER_TURBO_GROUP 128U
#define CLLM_WHISPER_TURBO_Q4_RECORD 66U
#define CLLM_WHISPER_TURBO_Q8_RECORD 130U
#define CLLM_WHISPER_TURBO_Q5_RECORD 82U

typedef struct {
    const float *f32;
    const unsigned char *q4;
    const unsigned char *q8;
    const unsigned char *q5;
    size_t rows;
    size_t columns;
} cllm_whisper_turbo_matrix;

static inline float cllm_whisper_turbo_bf16(const unsigned char *bytes)
{
    uint32_t bits = ((uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8U)) << 16U;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline size_t cllm_whisper_turbo_row_bytes(const cllm_whisper_turbo_matrix *matrix)
{
    const size_t groups = matrix->columns / CLLM_WHISPER_TURBO_GROUP;
    if (matrix->q4 != NULL) return groups * CLLM_WHISPER_TURBO_Q4_RECORD;
    if (matrix->q8 != NULL) return groups * CLLM_WHISPER_TURBO_Q8_RECORD;
    if (matrix->q5 != NULL) return groups * CLLM_WHISPER_TURBO_Q5_RECORD;
    return matrix->columns * sizeof(float);
}

#endif
