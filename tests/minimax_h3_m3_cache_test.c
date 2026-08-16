#include "minimax_h3_m3_cache.h"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(expression)                                                        \
    do {                                                                         \
        if (!(expression)) {                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,            \
                    #expression);                                                \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

int main(void) {
    minimax_h3_m3_cache_layout layout;
    minimax_h3_m3_cache_schedule schedule;
    size_t offset;
    size_t evaluation;
    size_t band;
    size_t modality;

    CHECK(minimax_h3_m3_cache_layout_make(15485u, &layout) == MINIMAX_H3_OK);
    CHECK(layout.hidden_size == 5376u);
    CHECK(layout.groups_per_vector == 84u);
    CHECK(layout.bytes_per_vector_per_branch == 3024u);
    CHECK(layout.bytes_per_layer == 93653280u);
    CHECK(layout.file_bytes == 4682664000u);
    CHECK(minimax_h3_m3_cache_vector_offset(&layout, 0u, 0u, 0u, &offset) ==
          MINIMAX_H3_OK && offset == 0u);
    CHECK(minimax_h3_m3_cache_vector_offset(&layout, 49u, 1u, 15484u,
                                             &offset) == MINIMAX_H3_OK);
    CHECK(offset + layout.bytes_per_vector_per_branch == layout.file_bytes);
    CHECK(minimax_h3_m3_cache_vector_offset(&layout, 50u, 0u, 0u, &offset) ==
          MINIMAX_H3_INVALID_ARGUMENT);
    CHECK(minimax_h3_m3_cache_layer_band(0u) == 0u);
    CHECK(minimax_h3_m3_cache_layer_band(49u) == 4u);
    CHECK(minimax_h3_m3_cache_layer_band(50u) == SIZE_MAX);

    memset(&schedule, 0, sizeof(schedule));
    for (evaluation = 0u; evaluation < 4u; ++evaluation) {
        for (band = 0u; band < 5u; ++band) {
            for (modality = 0u; modality < 3u; ++modality) {
                schedule.query_keep_per_mille[evaluation][band][modality] =
                    evaluation == 0u ? 1000u :
                    (uint16_t)(200u + evaluation * 100u);
                schedule.mlp_keep_per_mille[evaluation][band][modality] =
                    evaluation == 0u ? 1000u :
                    (uint16_t)(150u + evaluation * 150u);
            }
        }
    }
    CHECK(minimax_h3_m3_cache_schedule_validate(&schedule) == MINIMAX_H3_OK);
    schedule.query_keep_per_mille[2][1][MINIMAX_H3_VIDEO_TAG] = 250u;
    CHECK(minimax_h3_m3_cache_schedule_validate(&schedule) ==
          MINIMAX_H3_INVALID_ARGUMENT);
    schedule.query_keep_per_mille[2][1][MINIMAX_H3_VIDEO_TAG] = 400u;
    schedule.query_keep_per_mille[0][1][MINIMAX_H3_VIDEO_TAG] = 999u;
    CHECK(minimax_h3_m3_cache_schedule_validate(&schedule) ==
          MINIMAX_H3_INVALID_ARGUMENT);

    if (failures != 0) {
        fprintf(stderr, "%d MiniMax-H3 M3 cache checks failed\n", failures);
        return 1;
    }
    puts("MiniMax-H3 M3 branch-cache layout and schedule: PASS");
    return 0;
}
