#ifndef MINIMAX_H3_H
#define MINIMAX_H3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MINIMAX_H3_VIDEO_TAG = 0,
    MINIMAX_H3_TEXT_TAG = 1,
    MINIMAX_H3_AUDIO_TAG = 2,
    MINIMAX_H3_MODALITY_COUNT = 3,
    MINIMAX_H3_FPS = 24,
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40,
    MINIMAX_H3_AUDIO_CHANNELS = 2,
    MINIMAX_H3_VAE_SPATIAL_COMPRESSION = 16,
    MINIMAX_H3_VAE_FRAMES_PER_CHUNK = 17,
    MINIMAX_H3_VAE_LATENTS_PER_CHUNK = 5,
    MINIMAX_H3_VIDEO_PATCH_H = 2,
    MINIMAX_H3_VIDEO_PATCH_W = 2
};

typedef enum {
    MINIMAX_H3_OK = 0,
    MINIMAX_H3_INVALID_ARGUMENT = -1,
    MINIMAX_H3_INSUFFICIENT_CAPACITY = -2,
    MINIMAX_H3_OVERFLOW = -3
} minimax_h3_status;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t num_frames;
    uint32_t video_latent_frames;
    uint32_t latent_width;
    uint32_t latent_height;
    uint32_t audio_latent_frames;
    size_t text_rows;
    size_t rows_per_video_frame;
    size_t audio_rows;
    size_t video_rows;
    size_t sequence_rows;
} minimax_h3_geometry;

typedef struct {
    size_t text_start;
    size_t text_count;
    size_t audio_start;
    size_t audio_count;
    size_t video_start;
    size_t video_count;
    size_t sequence_rows;
} minimax_h3_t2va_layout;

typedef struct {
    size_t sigma_count;
    size_t evaluation_count;
} minimax_h3_schedule;

const char *minimax_h3_status_string(minimax_h3_status status);

minimax_h3_status minimax_h3_align_num_frames(uint32_t requested, uint32_t *aligned);

minimax_h3_status minimax_h3_video_latent_num_frames(uint32_t aligned_frames,
                                                     uint32_t *latent_frames);

minimax_h3_status minimax_h3_audio_latent_num_frames(uint32_t num_frames,
                                                     uint32_t *latent_frames);

/*
 * Build the exact T2VA tensor geometry for a caller-supplied canvas. H3 Base
 * requires both pixel axes to be divisible by 32: VAE /16, then patch /2.
 */
minimax_h3_status minimax_h3_geometry_init(minimax_h3_geometry *geometry,
                                           uint32_t width,
                                           uint32_t height,
                                           uint32_t requested_frames,
                                           size_t text_rows);

/*
 * Fill the packed [text | stereo audio | video] sequence used by T2VA.
 * position_ids is row-major [sequence_rows][3] in float64. Passing NULL for
 * text_tags assigns MINIMAX_H3_TEXT_TAG to every text row. No allocation is
 * performed; both output arrays belong to the caller.
 */
minimax_h3_status minimax_h3_build_t2va_layout(const minimax_h3_geometry *geometry,
                                               const uint8_t *text_tags,
                                               double *position_ids,
                                               uint8_t *token_tags,
                                               size_t capacity_rows,
                                               minimax_h3_t2va_layout *layout);

/*
 * Build MiniMax-H3's shifted rectified-flow grid. grid_points includes the
 * terminal zero; therefore evaluation_count is normally grid_points - 1.
 * Consecutive float32 collisions are removed as in the reference scheduler.
 */
minimax_h3_status minimax_h3_schedule_make(size_t grid_points,
                                           float shift,
                                           float *sigmas,
                                           size_t sigma_capacity,
                                           float *timesteps,
                                           size_t timestep_capacity,
                                           minimax_h3_schedule *schedule);

/*
 * Apply H3's flow shift to an explicit, strictly decreasing base-sigma list.
 * The list includes the terminal zero. This is the boundary used by external
 * step selectors such as the Turbo-4 ComfyUI `simple` schedule; it deliberately
 * remains separate from minimax_h3_schedule_make's evenly spaced grid.
 */
minimax_h3_status minimax_h3_schedule_from_base(const float *base_sigmas,
                                                size_t base_sigma_count,
                                                float shift,
                                                float *sigmas,
                                                size_t sigma_capacity,
                                                float *timesteps,
                                                size_t timestep_capacity,
                                                minimax_h3_schedule *schedule);

void minimax_h3_scale_noise(float *output,
                            const float *sample,
                            const float *noise,
                            size_t count,
                            float timestep);

/* One data-ward rectified-flow Euler update, evaluated in float32. */
minimax_h3_status minimax_h3_scheduler_step(float *sample,
                                            const float *model_output,
                                            size_t count,
                                            float timestep,
                                            float sigma,
                                            float sigma_next);

/*
 * Produce the two distinct row timesteps for a T2VA denoising call and the
 * per-row AdaLN table index: timestep_index * 3 + modality_tag.
 */
minimax_h3_status minimax_h3_build_t2va_row_plan(const minimax_h3_t2va_layout *layout,
                                                 const uint8_t *token_tags,
                                                 float video_timestep,
                                                 float audio_timestep,
                                                 float unique_timesteps[2],
                                                 size_t *unique_count,
                                                 uint8_t *row_timestep_indices,
                                                 uint8_t *adaln_indices,
                                                 size_t capacity_rows);

#ifdef __cplusplus
}
#endif

#endif
