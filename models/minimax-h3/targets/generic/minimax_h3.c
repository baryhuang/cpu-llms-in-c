#include "minimax_h3.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define MINIMAX_H3_ROPE_FRAME_RESCALE (5.0 / 3.0)
#define MINIMAX_H3_ROPE_SPATIAL_SCALE 32.0

static const uint8_t minimax_h3_rope_frames_per_latent[5] = {1, 4, 4, 4, 4};

static int checked_mul_size(size_t left, size_t right, size_t *result) {
    if (left != 0 && right > SIZE_MAX / left) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static int checked_add_size(size_t left, size_t right, size_t *result) {
    if (right > SIZE_MAX - left) {
        return 0;
    }
    *result = left + right;
    return 1;
}

const char *minimax_h3_status_string(minimax_h3_status status) {
    switch (status) {
        case MINIMAX_H3_OK:
            return "ok";
        case MINIMAX_H3_INVALID_ARGUMENT:
            return "invalid argument";
        case MINIMAX_H3_INSUFFICIENT_CAPACITY:
            return "insufficient capacity";
        case MINIMAX_H3_OVERFLOW:
            return "integer overflow";
        default:
            return "unknown MiniMax-H3 status";
    }
}

minimax_h3_status minimax_h3_align_num_frames(uint32_t requested, uint32_t *aligned) {
    uint32_t delta;

    if (aligned == NULL || requested == 0) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    delta = (MINIMAX_H3_VAE_LATENTS_PER_CHUNK + MINIMAX_H3_VAE_FRAMES_PER_CHUNK -
             requested % MINIMAX_H3_VAE_FRAMES_PER_CHUNK) %
            MINIMAX_H3_VAE_FRAMES_PER_CHUNK;
    if (requested > UINT32_MAX - delta) {
        return MINIMAX_H3_OVERFLOW;
    }
    *aligned = requested + delta;
    return MINIMAX_H3_OK;
}

minimax_h3_status minimax_h3_video_latent_num_frames(uint32_t aligned_frames,
                                                     uint32_t *latent_frames) {
    uint64_t chunks;
    uint64_t result;

    if (latent_frames == NULL || aligned_frames == 0 ||
        aligned_frames % MINIMAX_H3_VAE_FRAMES_PER_CHUNK !=
            MINIMAX_H3_VAE_LATENTS_PER_CHUNK) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    chunks = (aligned_frames - MINIMAX_H3_VAE_LATENTS_PER_CHUNK) /
             MINIMAX_H3_VAE_FRAMES_PER_CHUNK;
    result = chunks * MINIMAX_H3_VAE_LATENTS_PER_CHUNK + 2;
    if (result > UINT32_MAX) {
        return MINIMAX_H3_OVERFLOW;
    }
    *latent_frames = (uint32_t)result;
    return MINIMAX_H3_OK;
}

minimax_h3_status minimax_h3_audio_latent_num_frames(uint32_t num_frames,
                                                     uint32_t *latent_frames) {
    uint64_t numerator;
    uint64_t result;

    if (latent_frames == NULL || num_frames == 0) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    /* round(num_frames / 24 * 40) == round(num_frames * 5 / 3). */
    numerator = (uint64_t)num_frames * 5u;
    result = (numerator + 1u) / 3u;
    if (result > UINT32_MAX) {
        return MINIMAX_H3_OVERFLOW;
    }
    *latent_frames = (uint32_t)result;
    return MINIMAX_H3_OK;
}

minimax_h3_status minimax_h3_geometry_init(minimax_h3_geometry *geometry,
                                           uint32_t width,
                                           uint32_t height,
                                           uint32_t requested_frames,
                                           size_t text_rows) {
    minimax_h3_geometry result;
    minimax_h3_status status;
    size_t audio_rows;
    size_t video_rows;
    size_t sequence_rows;

    if (geometry == NULL || width == 0 || height == 0 || text_rows == 0 ||
        width % (MINIMAX_H3_VAE_SPATIAL_COMPRESSION * MINIMAX_H3_VIDEO_PATCH_W) != 0 ||
        height % (MINIMAX_H3_VAE_SPATIAL_COMPRESSION * MINIMAX_H3_VIDEO_PATCH_H) != 0) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    if ((double)width / (double)height < 0.25 || (double)width / (double)height > 4.0) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }

    memset(&result, 0, sizeof(result));
    result.width = width;
    result.height = height;
    result.latent_width = width / MINIMAX_H3_VAE_SPATIAL_COMPRESSION;
    result.latent_height = height / MINIMAX_H3_VAE_SPATIAL_COMPRESSION;
    result.text_rows = text_rows;

    status = minimax_h3_align_num_frames(requested_frames, &result.num_frames);
    if (status != MINIMAX_H3_OK) {
        return status;
    }
    status = minimax_h3_video_latent_num_frames(result.num_frames,
                                                &result.video_latent_frames);
    if (status != MINIMAX_H3_OK) {
        return status;
    }
    status = minimax_h3_audio_latent_num_frames(result.num_frames,
                                                &result.audio_latent_frames);
    if (status != MINIMAX_H3_OK) {
        return status;
    }

    if (!checked_mul_size((size_t)(result.latent_height / MINIMAX_H3_VIDEO_PATCH_H),
                          (size_t)(result.latent_width / MINIMAX_H3_VIDEO_PATCH_W),
                          &result.rows_per_video_frame) ||
        !checked_mul_size((size_t)result.audio_latent_frames,
                          MINIMAX_H3_AUDIO_CHANNELS,
                          &audio_rows) ||
        !checked_mul_size((size_t)result.video_latent_frames,
                          result.rows_per_video_frame,
                          &video_rows) ||
        !checked_add_size(text_rows, audio_rows, &sequence_rows) ||
        !checked_add_size(sequence_rows, video_rows, &sequence_rows)) {
        return MINIMAX_H3_OVERFLOW;
    }
    result.audio_rows = audio_rows;
    result.video_rows = video_rows;
    result.sequence_rows = sequence_rows;
    *geometry = result;
    return MINIMAX_H3_OK;
}

static double spatial_coordinate(uint32_t dimension,
                                 uint32_t patch,
                                 double square_root_area,
                                 uint32_t index) {
    double ratio = (double)dimension / square_root_area;
    double left = (1.0 - ratio) * 0.5;
    uint32_t count = dimension / patch;
    return (left + (double)index * ratio / (double)count) *
           MINIMAX_H3_ROPE_SPATIAL_SCALE;
}

minimax_h3_status minimax_h3_build_t2va_layout(const minimax_h3_geometry *geometry,
                                               const uint8_t *text_tags,
                                               double *position_ids,
                                               uint8_t *token_tags,
                                               size_t capacity_rows,
                                               minimax_h3_t2va_layout *layout) {
    minimax_h3_t2va_layout result;
    double square_root_area;
    double width_first;
    double width_last;
    double video_time;
    size_t row;
    size_t frame;
    uint32_t patch_rows;
    uint32_t patch_columns;

    if (geometry == NULL || position_ids == NULL || token_tags == NULL || layout == NULL) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    if (capacity_rows < geometry->sequence_rows) {
        return MINIMAX_H3_INSUFFICIENT_CAPACITY;
    }

    result.text_start = 0;
    result.text_count = geometry->text_rows;
    result.audio_start = result.text_count;
    result.audio_count = geometry->audio_rows;
    result.video_start = result.audio_start + result.audio_count;
    result.video_count = geometry->video_rows;
    result.sequence_rows = geometry->sequence_rows;

    memset(position_ids, 0, geometry->sequence_rows * 3u * sizeof(*position_ids));
    for (row = 0; row < result.text_count; ++row) {
        uint8_t tag = text_tags == NULL ? MINIMAX_H3_TEXT_TAG : text_tags[row];
        if (tag >= MINIMAX_H3_MODALITY_COUNT) {
            return MINIMAX_H3_INVALID_ARGUMENT;
        }
        position_ids[row * 3u] = (double)row;
        token_tags[row] = tag;
    }

    square_root_area = sqrt((double)geometry->latent_height * (double)geometry->latent_width);
    patch_rows = geometry->latent_height / MINIMAX_H3_VIDEO_PATCH_H;
    patch_columns = geometry->latent_width / MINIMAX_H3_VIDEO_PATCH_W;
    width_first = spatial_coordinate(geometry->latent_width,
                                     MINIMAX_H3_VIDEO_PATCH_W,
                                     square_root_area,
                                     0);
    width_last = spatial_coordinate(geometry->latent_width,
                                    MINIMAX_H3_VIDEO_PATCH_W,
                                    square_root_area,
                                    patch_columns - 1u);

    for (row = 0; row < result.audio_count; ++row) {
        size_t output_row = result.audio_start + row;
        size_t channel = row / geometry->audio_latent_frames;
        size_t audio_frame = row % geometry->audio_latent_frames;
        position_ids[output_row * 3u] = (double)geometry->text_rows + (double)audio_frame;
        position_ids[output_row * 3u + 2u] = channel == 0 ? width_first : width_last;
        token_tags[output_row] = MINIMAX_H3_AUDIO_TAG;
    }

    video_time = (double)geometry->text_rows;
    for (frame = 0; frame < geometry->video_latent_frames; ++frame) {
        uint32_t patch_y;
        uint32_t patch_x;
        for (patch_y = 0; patch_y < patch_rows; ++patch_y) {
            double y = spatial_coordinate(geometry->latent_height,
                                          MINIMAX_H3_VIDEO_PATCH_H,
                                          square_root_area,
                                          patch_y);
            for (patch_x = 0; patch_x < patch_columns; ++patch_x) {
                double x = spatial_coordinate(geometry->latent_width,
                                              MINIMAX_H3_VIDEO_PATCH_W,
                                              square_root_area,
                                              patch_x);
                size_t frame_row = (size_t)patch_y * patch_columns + patch_x;
                size_t output_row = result.video_start +
                                    frame * geometry->rows_per_video_frame + frame_row;
                position_ids[output_row * 3u] = video_time;
                position_ids[output_row * 3u + 1u] = y;
                position_ids[output_row * 3u + 2u] = x;
                token_tags[output_row] = MINIMAX_H3_VIDEO_TAG;
            }
        }
        video_time += MINIMAX_H3_ROPE_FRAME_RESCALE *
                      minimax_h3_rope_frames_per_latent[frame % 5u];
    }

    *layout = result;
    return MINIMAX_H3_OK;
}

minimax_h3_status minimax_h3_schedule_make(size_t grid_points,
                                           float shift,
                                           float *sigmas,
                                           size_t sigma_capacity,
                                           float *timesteps,
                                           size_t timestep_capacity,
                                           minimax_h3_schedule *schedule) {
    size_t input_index;
    size_t output_count = 0;

    if (grid_points < 2 || !isfinite(shift) || shift <= 0.0f || sigmas == NULL ||
        timesteps == NULL || schedule == NULL) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    if (sigma_capacity < grid_points || timestep_capacity < grid_points - 1u) {
        return MINIMAX_H3_INSUFFICIENT_CAPACITY;
    }

    for (input_index = 0; input_index < grid_points; ++input_index) {
        float base;
        float shifted;
        volatile float numerator;
        volatile float scaled_base;
        volatile float denominator;
        float step = -1.0f / (float)(grid_points - 1u);
        size_t halfway = grid_points / 2u;
        /* This mirrors ATen's float32 CPU linspace kernel. It evaluates the
         * first half from start and the second half from end so both endpoints
         * are exact. A double step or the algebraic 1-i/(n-1) form differs by
         * one ulp at some grid points. */
        if (input_index < halfway) {
            base = 1.0f + step * (float)input_index;
        } else {
            base = 0.0f - step * (float)(grid_points - input_index - 1u);
        }
        /* The reference performs multiply, multiply, add and divide as
         * separate tensor operations. Volatile temporaries prevent the C
         * compiler from contracting the denominator into an FMA, which changes
         * the final two shift-12 entries by one or two ulps on Apple clang. */
        numerator = shift * base;
        scaled_base = (shift - 1.0f) * base;
        denominator = 1.0f + scaled_base;
        shifted = numerator / denominator;
        if (output_count == 0 || shifted != sigmas[output_count - 1u]) {
            sigmas[output_count++] = shifted;
        }
    }
    if (output_count < 2 || sigmas[output_count - 1u] != 0.0f) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    for (input_index = 0; input_index + 1u < output_count; ++input_index) {
        timesteps[input_index] = 1.0f - sigmas[input_index];
    }
    schedule->sigma_count = output_count;
    schedule->evaluation_count = output_count - 1u;
    return MINIMAX_H3_OK;
}

minimax_h3_status minimax_h3_schedule_from_base(const float *base_sigmas,
                                                size_t base_sigma_count,
                                                float shift,
                                                float *sigmas,
                                                size_t sigma_capacity,
                                                float *timesteps,
                                                size_t timestep_capacity,
                                                minimax_h3_schedule *schedule) {
    size_t index;

    if (base_sigmas == NULL || base_sigma_count < 2u || !isfinite(shift) ||
        shift <= 0.0f || sigmas == NULL || timesteps == NULL || schedule == NULL) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    if (sigma_capacity < base_sigma_count ||
        timestep_capacity < base_sigma_count - 1u) {
        return MINIMAX_H3_INSUFFICIENT_CAPACITY;
    }
    if (base_sigmas[0] != 1.0f ||
        base_sigmas[base_sigma_count - 1u] != 0.0f) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }

    for (index = 0; index < base_sigma_count; ++index) {
        float base = base_sigmas[index];
        float shifted;
        volatile float numerator;
        volatile float scaled_base;
        volatile float denominator;

        if (!isfinite(base) || base < 0.0f || base > 1.0f ||
            (index > 0u && base >= base_sigmas[index - 1u])) {
            return MINIMAX_H3_INVALID_ARGUMENT;
        }
        numerator = shift * base;
        scaled_base = (shift - 1.0f) * base;
        denominator = 1.0f + scaled_base;
        shifted = numerator / denominator;
        sigmas[index] = shifted;
        if (index + 1u < base_sigma_count) {
            timesteps[index] = 1.0f - shifted;
        }
    }

    schedule->sigma_count = base_sigma_count;
    schedule->evaluation_count = base_sigma_count - 1u;
    return MINIMAX_H3_OK;
}

void minimax_h3_scale_noise(float *output,
                            const float *sample,
                            const float *noise,
                            size_t count,
                            float timestep) {
    size_t index;
    if (output == NULL || sample == NULL || noise == NULL) {
        return;
    }
    for (index = 0; index < count; ++index) {
        output[index] = timestep * sample[index] + (1.0f - timestep) * noise[index];
    }
}

minimax_h3_status minimax_h3_scheduler_step(float *sample,
                                            const float *model_output,
                                            size_t count,
                                            float timestep,
                                            float sigma,
                                            float sigma_next) {
    float ratio;
    float sigma_from_timestep;
    size_t index;

    if (sample == NULL || model_output == NULL || !isfinite(timestep) ||
        !isfinite(sigma) || !isfinite(sigma_next) || timestep < 0.0f ||
        timestep > 1.0f || sigma <= 0.0f || sigma_next < 0.0f ||
        sigma_next >= sigma) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    ratio = sigma_next / sigma;
    sigma_from_timestep = 1.0f - timestep;
    for (index = 0; index < count; ++index) {
        float denoised = sample[index] + sigma_from_timestep * model_output[index];
        sample[index] = ratio * sample[index] + (1.0f - ratio) * denoised;
    }
    return MINIMAX_H3_OK;
}

minimax_h3_status minimax_h3_build_t2va_row_plan(const minimax_h3_t2va_layout *layout,
                                                 const uint8_t *token_tags,
                                                 float video_timestep,
                                                 float audio_timestep,
                                                 float unique_timesteps[2],
                                                 size_t *unique_count,
                                                 uint8_t *row_timestep_indices,
                                                 uint8_t *adaln_indices,
                                                 size_t capacity_rows) {
    uint8_t video_index;
    uint8_t audio_index;
    size_t row;

    if (layout == NULL || token_tags == NULL || unique_timesteps == NULL ||
        unique_count == NULL || row_timestep_indices == NULL || adaln_indices == NULL ||
        !isfinite(video_timestep) || !isfinite(audio_timestep)) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    if (capacity_rows < layout->sequence_rows) {
        return MINIMAX_H3_INSUFFICIENT_CAPACITY;
    }

    if (video_timestep == audio_timestep) {
        unique_timesteps[0] = video_timestep;
        *unique_count = 1;
        video_index = 0;
        audio_index = 0;
    } else if (video_timestep < audio_timestep) {
        unique_timesteps[0] = video_timestep;
        unique_timesteps[1] = audio_timestep;
        *unique_count = 2;
        video_index = 0;
        audio_index = 1;
    } else {
        unique_timesteps[0] = audio_timestep;
        unique_timesteps[1] = video_timestep;
        *unique_count = 2;
        video_index = 1;
        audio_index = 0;
    }

    for (row = 0; row < layout->sequence_rows; ++row) {
        uint8_t timestep_index;
        if (token_tags[row] >= MINIMAX_H3_MODALITY_COUNT) {
            return MINIMAX_H3_INVALID_ARGUMENT;
        }
        timestep_index = row >= layout->audio_start && row < layout->video_start
                             ? audio_index
                             : video_index;
        row_timestep_indices[row] = timestep_index;
        adaln_indices[row] = (uint8_t)(timestep_index * MINIMAX_H3_MODALITY_COUNT +
                                       token_tags[row]);
    }
    return MINIMAX_H3_OK;
}
