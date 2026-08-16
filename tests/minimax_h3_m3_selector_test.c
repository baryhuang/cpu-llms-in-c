#include "minimax_h3.h"
#include "minimax_h3_m3_selector.h"
#include "minimax_h3_m3_tree.h"

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
    minimax_h3_m3_tree_plan plan;
    minimax_h3_m3_selector_weights weights = {0.5f, 1.0f, 0.25f, 0.01f};
    minimax_h3_m3_tree_node *nodes;
    double *positions;
    uint8_t *tags;
    float *lse;
    float *delta;
    float *priority;
    uint16_t *age;
    uint8_t *selected;
    uint32_t *indices;
    size_t node_count;
    size_t selected_count;
    size_t selected_rows;
    size_t leaves_per_frame;

    CHECK(minimax_h3_geometry_init(&geometry, 864u, 480u, 124u, 86u) ==
          MINIMAX_H3_OK);
    positions = malloc(geometry.sequence_rows * 3u * sizeof(*positions));
    tags = malloc(geometry.sequence_rows * sizeof(*tags));
    CHECK(positions != NULL && tags != NULL);
    CHECK(minimax_h3_build_t2va_layout(&geometry, NULL, positions, tags,
                                        geometry.sequence_rows, &layout) ==
          MINIMAX_H3_OK);
    CHECK(minimax_h3_m3_tree_node_count(&geometry, &node_count) ==
          MINIMAX_H3_OK);
    nodes = calloc(node_count, sizeof(*nodes));
    CHECK(nodes != NULL);
    CHECK(minimax_h3_m3_tree_plan_make(&geometry, &layout, nodes, node_count,
                                       &plan) == MINIMAX_H3_OK);
    lse = malloc(plan.leaf_count * sizeof(*lse));
    delta = malloc(plan.leaf_count * sizeof(*delta));
    priority = malloc(plan.leaf_count * sizeof(*priority));
    age = malloc(plan.leaf_count * sizeof(*age));
    selected = malloc(plan.leaf_count * sizeof(*selected));
    indices = malloc(plan.leaf_count * sizeof(*indices));
    CHECK(lse != NULL && delta != NULL && priority != NULL && age != NULL &&
          selected != NULL && indices != NULL);
    leaves_per_frame = plan.leaf_count / plan.frame_node_count;
    for (size_t leaf = 0u; leaf < plan.leaf_count; ++leaf) {
        size_t frame = leaf / leaves_per_frame;
        size_t spatial = leaf % leaves_per_frame;
        lse[leaf] = spatial == frame % leaves_per_frame ? 10.0f : 0.1f;
        delta[leaf] = (float)((leaf * 7u) % 13u) * 0.01f;
        age[leaf] = (uint16_t)(leaf % 5u);
    }
    CHECK(minimax_h3_m3_select_video_leaves(
              nodes, &plan, lse, delta, age, 180u, &weights, priority,
              selected, indices, plan.leaf_count, &selected_count,
              &selected_rows) == MINIMAX_H3_OK);
    CHECK(selected_count == 54u);
    CHECK(selected_rows > 0u && selected_rows <= geometry.video_rows);
    for (size_t index = 1u; index < selected_count; ++index)
        CHECK(indices[index - 1u] < indices[index]);
    for (size_t frame = 0u; frame < plan.frame_node_count; ++frame) {
        size_t retained = 0u;
        for (size_t spatial = 0u; spatial < leaves_per_frame; ++spatial)
            retained += selected[frame * leaves_per_frame + spatial] != 0u;
        CHECK(retained != 0u);
    }

    free(indices);
    free(selected);
    free(age);
    free(priority);
    free(delta);
    free(lse);
    free(nodes);
    free(tags);
    free(positions);
    if (failures != 0) {
        fprintf(stderr, "%d MiniMax-H3 selector checks failed\n", failures);
        return 1;
    }
    puts("MiniMax-H3 M3 temporal leaf selector: PASS");
    return 0;
}
