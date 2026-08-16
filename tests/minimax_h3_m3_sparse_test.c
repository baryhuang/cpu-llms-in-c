#include "minimax_h3.h"
#include "minimax_h3_m3_sparse.h"

#include <stdio.h>
#include <stdlib.h>

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
    minimax_h3_geometry geometry;
    minimax_h3_t2va_layout layout;
    minimax_h3_m3_cache_schedule schedule;
    minimax_h3_m3_sparse_work dense;
    minimax_h3_m3_sparse_work sparse;
    double *position_ids;
    uint8_t *token_tags;

    CHECK(minimax_h3_geometry_init(&geometry, 864u, 480u, 124u, 86u) ==
          MINIMAX_H3_OK);
    position_ids = malloc(geometry.sequence_rows * 3u * sizeof(*position_ids));
    token_tags = malloc(geometry.sequence_rows * sizeof(*token_tags));
    CHECK(position_ids != NULL && token_tags != NULL);
    CHECK(minimax_h3_build_t2va_layout(&geometry, NULL, position_ids, token_tags,
                                        geometry.sequence_rows, &layout) ==
          MINIMAX_H3_OK);
    minimax_h3_m3_cache_schedule_seed(&schedule);
    CHECK(minimax_h3_m3_cache_schedule_validate(&schedule) == MINIMAX_H3_OK);
    CHECK(minimax_h3_m3_sparse_work_make(
              &layout, &schedule, 0u, 2u,
              MINIMAX_H3_M3_NATIVE_SUMMARY_ROWS, &dense) == MINIMAX_H3_OK);
    CHECK(dense.total_query_rows == 15485u);
    CHECK(dense.total_mlp_rows == 15485u);
    CHECK(dense.kv_projection_rows == 15485u);
    CHECK(dense.summary_rows == 0u);
    CHECK(dense.dense_projection_macs == UINT64_C(5967170764800));
    CHECK(dense.sparse_projection_macs == dense.dense_projection_macs);

    CHECK(minimax_h3_m3_sparse_work_make(
              &layout, &schedule, 1u, 2u,
              MINIMAX_H3_M3_NATIVE_SUMMARY_ROWS, &sparse) == MINIMAX_H3_OK);
    CHECK(sparse.query_rows[MINIMAX_H3_VIDEO_TAG] == 2698u);
    CHECK(sparse.query_rows[MINIMAX_H3_TEXT_TAG] == 86u);
    CHECK(sparse.query_rows[MINIMAX_H3_AUDIO_TAG] == 270u);
    CHECK(sparse.total_query_rows == 3054u);
    CHECK(sparse.total_mlp_rows == 2583u);
    CHECK(sparse.kv_projection_rows == 3420u);
    CHECK(sparse.sparse_projection_macs < sparse.dense_projection_macs / 4u);

    free(token_tags);
    free(position_ids);

    if (failures != 0) {
        fprintf(stderr, "%d MiniMax-H3 sparse-plan checks failed\n", failures);
        return 1;
    }
    puts("MiniMax-H3 M3 sparse projection plan: PASS");
    return 0;
}
