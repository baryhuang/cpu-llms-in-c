#include "minimax_h3.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition)                                                               \
    do {                                                                               \
        if (!(condition)) {                                                            \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);      \
            failures++;                                                                \
        }                                                                              \
    } while (0)

static int close_double(double left, double right, double tolerance) {
    return fabs(left - right) <= tolerance;
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void test_geometry(void) {
    minimax_h3_geometry geometry;
    uint32_t value = 0;

    CHECK(minimax_h3_align_num_frames(1, &value) == MINIMAX_H3_OK && value == 5);
    CHECK(minimax_h3_align_num_frames(5, &value) == MINIMAX_H3_OK && value == 5);
    CHECK(minimax_h3_align_num_frames(6, &value) == MINIMAX_H3_OK && value == 22);
    CHECK(minimax_h3_align_num_frames(124, &value) == MINIMAX_H3_OK && value == 124);
    CHECK(minimax_h3_video_latent_num_frames(124, &value) == MINIMAX_H3_OK && value == 37);
    CHECK(minimax_h3_video_latent_num_frames(123, &value) == MINIMAX_H3_INVALID_ARGUMENT);
    CHECK(minimax_h3_audio_latent_num_frames(124, &value) == MINIMAX_H3_OK && value == 207);

    CHECK(minimax_h3_geometry_init(&geometry, 1344, 768, 124, 4096) == MINIMAX_H3_OK);
    CHECK(geometry.num_frames == 124);
    CHECK(geometry.video_latent_frames == 37);
    CHECK(geometry.latent_width == 84 && geometry.latent_height == 48);
    CHECK(geometry.audio_latent_frames == 207);
    CHECK(geometry.rows_per_video_frame == 1008);
    CHECK(geometry.audio_rows == 414);
    CHECK(geometry.video_rows == 37296);
    CHECK(geometry.sequence_rows == 41806);
    CHECK(minimax_h3_geometry_init(&geometry, 1330, 768, 124, 10) ==
          MINIMAX_H3_INVALID_ARGUMENT);

    /* MacOS-H3-Speedrun's published attention workload: 14,985 video,
     * 414 audio and 86 text rows. Keep this mapping pinned for comparison. */
    CHECK(minimax_h3_geometry_init(&geometry, 864, 480, 124, 86) == MINIMAX_H3_OK);
    CHECK(geometry.rows_per_video_frame == 405);
    CHECK(geometry.video_rows == 14985);
    CHECK(geometry.audio_rows == 414);
    CHECK(geometry.sequence_rows == 15485);
}

static void test_layout_and_row_plan(void) {
    minimax_h3_geometry geometry;
    minimax_h3_t2va_layout layout;
    double *positions;
    uint8_t *tags;
    uint8_t *row_timesteps;
    uint8_t *adaln_indices;
    float unique[2];
    size_t unique_count = 0;
    size_t last_audio;
    size_t second_video;

    CHECK(minimax_h3_geometry_init(&geometry, 32, 32, 5, 2) == MINIMAX_H3_OK);
    CHECK(geometry.video_latent_frames == 2);
    CHECK(geometry.audio_latent_frames == 8);
    CHECK(geometry.sequence_rows == 20);

    positions = (double *)calloc(geometry.sequence_rows * 3u, sizeof(*positions));
    tags = (uint8_t *)calloc(geometry.sequence_rows, sizeof(*tags));
    row_timesteps = (uint8_t *)calloc(geometry.sequence_rows, sizeof(*row_timesteps));
    adaln_indices = (uint8_t *)calloc(geometry.sequence_rows, sizeof(*adaln_indices));
    CHECK(positions != NULL && tags != NULL && row_timesteps != NULL && adaln_indices != NULL);
    if (positions == NULL || tags == NULL || row_timesteps == NULL || adaln_indices == NULL) {
        free(positions);
        free(tags);
        free(row_timesteps);
        free(adaln_indices);
        return;
    }

    CHECK(minimax_h3_build_t2va_layout(&geometry, NULL, positions, tags,
                                       geometry.sequence_rows, &layout) == MINIMAX_H3_OK);
    CHECK(layout.text_start == 0 && layout.text_count == 2);
    CHECK(layout.audio_start == 2 && layout.audio_count == 16);
    CHECK(layout.video_start == 18 && layout.video_count == 2);
    CHECK(tags[0] == MINIMAX_H3_TEXT_TAG && tags[1] == MINIMAX_H3_TEXT_TAG);
    CHECK(close_double(positions[0], 0.0, 0.0));
    CHECK(close_double(positions[3], 1.0, 0.0));
    CHECK(tags[layout.audio_start] == MINIMAX_H3_AUDIO_TAG);
    CHECK(close_double(positions[layout.audio_start * 3u], 2.0, 0.0));
    last_audio = layout.video_start - 1u;
    CHECK(close_double(positions[last_audio * 3u], 9.0, 0.0));
    CHECK(tags[layout.video_start] == MINIMAX_H3_VIDEO_TAG);
    CHECK(close_double(positions[layout.video_start * 3u], 2.0, 0.0));
    second_video = layout.video_start + 1u;
    CHECK(close_double(positions[second_video * 3u], 2.0 + 5.0 / 3.0, 1e-14));
    CHECK(close_double(positions[second_video * 3u + 1u], 0.0, 0.0));
    CHECK(close_double(positions[second_video * 3u + 2u], 0.0, 0.0));

    CHECK(minimax_h3_build_t2va_row_plan(&layout, tags, 0.25f, 0.5f, unique,
                                         &unique_count, row_timesteps, adaln_indices,
                                         geometry.sequence_rows) == MINIMAX_H3_OK);
    CHECK(unique_count == 2 && unique[0] == 0.25f && unique[1] == 0.5f);
    CHECK(row_timesteps[0] == 0 && adaln_indices[0] == 1);
    CHECK(row_timesteps[layout.audio_start] == 1 &&
          adaln_indices[layout.audio_start] == 5);
    CHECK(row_timesteps[layout.video_start] == 0 &&
          adaln_indices[layout.video_start] == 0);

    free(positions);
    free(tags);
    free(row_timesteps);
    free(adaln_indices);
}

static void test_scheduler(void) {
    static const uint32_t expected_video_sigma_bits[20] = {
        0x3f800000u, 0x3f7ed1feu, 0x3f7d83bbu, 0x3f7c0fc0u, 0x3f7a6f4fu,
        0x3f7899e5u, 0x3f7684beu, 0x3f7421e8u, 0x3f715f16u, 0x3f6e23b8u,
        0x3f6a4e1au, 0x3f65aea7u, 0x3f600000u, 0x3f58d8d8u, 0x3f4f914cu,
        0x3f430c30u, 0x3f313b13u, 0x3f15da8au, 0x3ecccccdu, 0x00000000u
    };
    static const uint32_t expected_audio_sigma_bits[20] = {
        0x3f800000u, 0x3f7b5871u, 0x3f7656f2u, 0x3f70f0f0u, 0x3f6b1a20u,
        0x3f64c415u, 0x3f5ddddeu, 0x3f565359u, 0x3f4e0c7eu, 0x3f44ec4fu,
        0x3f3acf92u, 0x3f2f8af8u, 0x3f22e8b9u, 0x3f14a529u, 0x3f0469efu,
        0x3ee38e38u, 0x3eb851ebu, 0x3e8590b2u, 0x3e124925u, 0x00000000u
    };
    float video_sigmas[20];
    float video_timesteps[19];
    float audio_sigmas[20];
    float audio_timesteps[19];
    minimax_h3_schedule video;
    minimax_h3_schedule audio;
    float sample[2] = {2.0f, -1.0f};
    float velocity[2] = {3.0f, 4.0f};
    float clean[2] = {10.0f, 20.0f};
    float noise[2] = {-2.0f, 2.0f};
    float mixed[2];
    float expected0;
    float expected1;
    size_t index;

    CHECK(minimax_h3_schedule_make(20, 12.0f, video_sigmas, 20,
                                   video_timesteps, 19, &video) == MINIMAX_H3_OK);
    CHECK(minimax_h3_schedule_make(20, 3.0f, audio_sigmas, 20,
                                   audio_timesteps, 19, &audio) == MINIMAX_H3_OK);
    CHECK(video.sigma_count == 20 && video.evaluation_count == 19);
    CHECK(audio.sigma_count == 20 && audio.evaluation_count == 19);
    CHECK(video_sigmas[0] == 1.0f && video_sigmas[19] == 0.0f);
    CHECK(audio_sigmas[0] == 1.0f && audio_sigmas[19] == 0.0f);
    for (index = 0; index < 20; ++index) {
        CHECK(float_bits(video_sigmas[index]) == expected_video_sigma_bits[index]);
        CHECK(float_bits(audio_sigmas[index]) == expected_audio_sigma_bits[index]);
    }
    CHECK(video_timesteps[0] == 0.0f && audio_timesteps[0] == 0.0f);
    CHECK(close_double(video_timesteps[18], 0.6, 1e-6));
    CHECK(close_double(audio_timesteps[18], 6.0 / 7.0, 1e-6));

    expected0 = (video_sigmas[1] / video_sigmas[0]) * sample[0] +
                (1.0f - video_sigmas[1] / video_sigmas[0]) *
                    (sample[0] + (1.0f - video_timesteps[0]) * velocity[0]);
    expected1 = (video_sigmas[1] / video_sigmas[0]) * sample[1] +
                (1.0f - video_sigmas[1] / video_sigmas[0]) *
                    (sample[1] + (1.0f - video_timesteps[0]) * velocity[1]);
    CHECK(minimax_h3_scheduler_step(sample, velocity, 2, video_timesteps[0],
                                    video_sigmas[0], video_sigmas[1]) == MINIMAX_H3_OK);
    CHECK(sample[0] == expected0 && sample[1] == expected1);

    minimax_h3_scale_noise(mixed, clean, noise, 2, 0.25f);
    CHECK(mixed[0] == 1.0f && mixed[1] == 6.5f);
}

static void test_turbo4_scheduler(void) {
    static const float base_sigmas[5] = {1.0f, 0.75f, 0.5f, 0.25f, 0.0f};
    static const uint32_t expected_video_sigma_bits[5] = {
        0x3f800000u, 0x3f7914c2u, 0x3f6c4ec5u, 0x3f4ccccdu, 0x00000000u
    };
    static const uint32_t expected_audio_sigma_bits[5] = {
        0x3f800000u, 0x3f666666u, 0x3f400000u, 0x3f000000u, 0x00000000u
    };
    static const uint32_t expected_video_timestep_bits[4] = {
        0x00000000u, 0x3cdd67c0u, 0x3d9d89d8u, 0x3e4cccccu
    };
    static const uint32_t expected_audio_timestep_bits[4] = {
        0x00000000u, 0x3dccccd0u, 0x3e800000u, 0x3f000000u
    };
    float video_sigmas[5];
    float audio_sigmas[5];
    float video_timesteps[4];
    float audio_timesteps[4];
    minimax_h3_schedule video;
    minimax_h3_schedule audio;
    size_t index;

    CHECK(minimax_h3_schedule_from_base(base_sigmas, 5, 12.0f, video_sigmas, 5,
                                        video_timesteps, 4, &video) == MINIMAX_H3_OK);
    CHECK(minimax_h3_schedule_from_base(base_sigmas, 5, 3.0f, audio_sigmas, 5,
                                        audio_timesteps, 4, &audio) == MINIMAX_H3_OK);
    CHECK(video.sigma_count == 5 && video.evaluation_count == 4);
    CHECK(audio.sigma_count == 5 && audio.evaluation_count == 4);
    for (index = 0; index < 5; ++index) {
        CHECK(float_bits(video_sigmas[index]) == expected_video_sigma_bits[index]);
        CHECK(float_bits(audio_sigmas[index]) == expected_audio_sigma_bits[index]);
    }
    for (index = 0; index < 4; ++index) {
        CHECK(float_bits(video_timesteps[index]) ==
              expected_video_timestep_bits[index]);
        CHECK(float_bits(audio_timesteps[index]) ==
              expected_audio_timestep_bits[index]);
    }

    CHECK(minimax_h3_schedule_from_base(base_sigmas, 5, 12.0f, video_sigmas, 4,
                                        video_timesteps, 4, &video) ==
          MINIMAX_H3_INSUFFICIENT_CAPACITY);
    {
        static const float invalid_base[5] = {1.0f, 0.75f, 0.75f, 0.25f, 0.0f};
        CHECK(minimax_h3_schedule_from_base(invalid_base, 5, 12.0f,
                                            video_sigmas, 5, video_timesteps, 4,
                                            &video) == MINIMAX_H3_INVALID_ARGUMENT);
    }
}

int main(void) {
    test_geometry();
    test_layout_and_row_plan();
    test_scheduler();
    test_turbo4_scheduler();
    if (failures != 0) {
        fprintf(stderr, "%d MiniMax-H3 checks failed\n", failures);
        return 1;
    }
    puts("MiniMax-H3 generic geometry, layout and scheduler: PASS");
    return 0;
}
