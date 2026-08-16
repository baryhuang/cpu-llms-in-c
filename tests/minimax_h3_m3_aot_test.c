#include "minimax_h3.h"
#include "minimax_h3_m3_aot.h"

#include <stdio.h>

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
    static const float base_sigmas[5] = {1.0f, 0.75f, 0.5f, 0.25f, 0.0f};
    float video_sigmas[5];
    float audio_sigmas[5];
    float video_timesteps[4];
    float audio_timesteps[4];
    minimax_h3_schedule video;
    minimax_h3_schedule audio;
    minimax_h3_m3_aot_plan plan;
    size_t evaluation;

    CHECK(minimax_h3_schedule_from_base(base_sigmas, 5, 12.0f, video_sigmas, 5,
                                        video_timesteps, 4, &video) == MINIMAX_H3_OK);
    CHECK(minimax_h3_schedule_from_base(base_sigmas, 5, 3.0f, audio_sigmas, 5,
                                        audio_timesteps, 4, &audio) == MINIMAX_H3_OK);
    CHECK(minimax_h3_m3_turbo4_aot_plan_make(video_timesteps, audio_timesteps,
                                             &plan) == MINIMAX_H3_OK);

    CHECK(plan.header_bytes == 16384u);
    CHECK(plan.modulation_offset == 16384u);
    CHECK(plan.modulation_bytes == 38707200u);
    CHECK(plan.final_norm_offset == 38731776u);
    CHECK(plan.final_norm_bytes == 150528u);
    CHECK(plan.file_bytes == 38895616u);
    CHECK(plan.file_bytes < 38u * 1024u * 1024u);

    for (evaluation = 0u; evaluation < 4u; ++evaluation) {
        uint8_t video_slot = plan.evaluation_modulation_slot[evaluation]
                                                               [MINIMAX_H3_VIDEO_TAG];
        uint8_t text_slot = plan.evaluation_modulation_slot[evaluation]
                                                              [MINIMAX_H3_TEXT_TAG];
        uint8_t audio_slot = plan.evaluation_modulation_slot[evaluation]
                                                               [MINIMAX_H3_AUDIO_TAG];
        CHECK(video_slot == evaluation * 3u);
        CHECK(text_slot == evaluation * 3u + 1u);
        CHECK(audio_slot == evaluation * 3u + 2u);
        CHECK(plan.modulation_slots[video_slot].timestep ==
              video_timesteps[evaluation]);
        CHECK(plan.modulation_slots[text_slot].timestep ==
              video_timesteps[evaluation]);
        CHECK(plan.modulation_slots[audio_slot].timestep ==
              audio_timesteps[evaluation]);
    }
    CHECK(plan.evaluation_final_slot[0][0] == 0u);
    CHECK(plan.evaluation_final_slot[0][1] == 0u);
    CHECK(plan.evaluation_final_slot[1][0] == 1u);
    CHECK(plan.evaluation_final_slot[1][1] == 2u);
    CHECK(plan.evaluation_final_slot[3][0] == 5u);
    CHECK(plan.evaluation_final_slot[3][1] == 6u);

    if (failures != 0) {
        fprintf(stderr, "%d MiniMax-H3 M3 AOT checks failed\n", failures);
        return 1;
    }
    puts("MiniMax-H3 M3 Turbo-4 AOT layout: PASS");
    return 0;
}
