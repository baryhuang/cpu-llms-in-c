#include "minimax_h3_m3_e2e.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void json_string(const char *value) {
    putchar('"');
    for (const unsigned char *cursor = (const unsigned char *)value;
         *cursor != '\0'; ++cursor) {
        if (*cursor == '"' || *cursor == '\\') putchar('\\');
        if (*cursor == '\n') fputs("\\n", stdout);
        else if (*cursor >= 0x20u) putchar(*cursor);
    }
    putchar('"');
}

int main(int argc, char **argv) {
    if (argc >= 4 && argc <= 6 &&
        strcmp(argv[1], "--image-vae-smoke") == 0) {
        minimax_h3_m3_e2e_options smoke_options = {
            .width = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 128u,
            .height = argc > 5 ? (uint32_t)strtoul(argv[5], NULL, 10) : 128u,
            .frames = 22u,
            .seed = UINT64_C(42),
            .prompt = "[conditioning image VAE]",
            .first_image_path = argv[2],
            .tokenizer_image_path = "build/minimax-h3-tokenizer.h3tok",
            .metallib_path = "build/minimax-h3-m3-attention.metallib",
            .output_directory = argv[3],
        };
        minimax_h3_m3_e2e_result smoke_result;
        char smoke_error[1024];
        int smoke_status = minimax_h3_m3_run_image_vae_smoke(
            &smoke_options, &smoke_result, smoke_error, sizeof(smoke_error));
        if (smoke_status != 0) {
            fprintf(stderr, "MiniMax-H3 image VAE smoke failed: %s\n",
                    smoke_error);
            return smoke_status;
        }
        printf("{\"status\":\"image-vae-smoke-passed\","
               "\"geometry\":{\"width\":%u,\"height\":%u},"
               "\"image_encode_seconds\":%.6f,"
               "\"peak_footprint_bytes\":%zu,\"latent_path\":",
               smoke_options.width, smoke_options.height,
               smoke_result.image_encode_seconds,
               smoke_result.peak_footprint_bytes);
        json_string(smoke_result.output_path);
        fputs("}\n", stdout);
        return 0;
    }
    const char *first_image = NULL;
    const char *last_image = NULL;
    int argument = 1;
    while (argument < argc &&
           (strcmp(argv[argument], "--first-image") == 0 ||
            strcmp(argv[argument], "--last-image") == 0)) {
        if (argument + 1 >= argc) {
            fprintf(stderr, "%s requires a path\n", argv[argument]);
            return 2;
        }
        if (strcmp(argv[argument], "--first-image") == 0)
            first_image = argv[argument + 1];
        else
            last_image = argv[argument + 1];
        argument += 2;
    }
    int remaining = argc - argument;
    int downstream_smoke = remaining >= 1 &&
                           strcmp(argv[argument], "--downstream-smoke") == 0;
    int video_vae_smoke = remaining >= 1 &&
                          strcmp(argv[argument], "--video-vae-smoke") == 0;
    if ((downstream_smoke && remaining != 2) ||
        (video_vae_smoke && (remaining < 2 || remaining > 5)) ||
        (!downstream_smoke && !video_vae_smoke &&
         (remaining < 2 || remaining > 6)) ||
        ((downstream_smoke || video_vae_smoke) &&
         (first_image != NULL || last_image != NULL))) {
        fprintf(stderr,
                "usage: %s [--first-image PATH] [--last-image PATH] "
                "PROMPT OUTPUT_DIR [WIDTH HEIGHT FRAMES SEED]\n"
                "       %s --downstream-smoke OUTPUT_DIR\n"
                "       %s --video-vae-smoke OUTPUT_DIR "
                "[WIDTH HEIGHT FRAMES]\n"
                "       %s --image-vae-smoke IMAGE OUTPUT_DIR "
                "[WIDTH HEIGHT]\n",
                argv[0], argv[0],
                argv[0], argv[0]);
        return 2;
    }
    minimax_h3_m3_e2e_options options = {
        .width = !downstream_smoke && remaining > 2
                     ? (uint32_t)strtoul(argv[argument + 2], NULL, 10)
                     : (video_vae_smoke ? 256u : 32u),
        .height = !downstream_smoke && remaining > 3
                      ? (uint32_t)strtoul(argv[argument + 3], NULL, 10)
                      : (video_vae_smoke ? 256u : 32u),
        .frames = !downstream_smoke && remaining > 4 ?
            (uint32_t)strtoul(argv[argument + 4], NULL, 10) : 22u,
        .seed = !downstream_smoke && !video_vae_smoke && remaining > 5 ?
            strtoull(argv[argument + 5], NULL, 10) : UINT64_C(42),
        .prompt = downstream_smoke
                      ? "[fixed conditioner state]"
                      : (video_vae_smoke ? "[deterministic VAE latent]"
                                         : argv[argument]),
        .first_image_path = first_image,
        .last_image_path = last_image,
        .tokenizer_image_path = "build/minimax-h3-tokenizer.h3tok",
        .metallib_path = "build/minimax-h3-m3-attention.metallib",
        .output_directory = argv[argument + 1],
    };
    minimax_h3_m3_e2e_result result;
    char error[1024];
    int status = downstream_smoke
                     ? minimax_h3_m3_run_downstream_smoke(
                           &options, &result, error, sizeof(error))
                     : (video_vae_smoke
                            ? minimax_h3_m3_run_video_vae_smoke(
                                  &options, &result, error, sizeof(error))
                            : minimax_h3_m3_run_e2e(
                                  &options, &result, error, sizeof(error)));
    if (status != 0) {
        fprintf(stderr, "MiniMax-H3 N-to-N failed: %s\n", error);
        return status;
    }
    printf("{\n  \"status\": \"%s\",\n  \"prompt\": ",
           downstream_smoke
               ? "downstream-smoke-passed"
               : (video_vae_smoke ? "video-vae-smoke-passed" : "passed"));
    json_string(options.prompt);
    printf(",\n  \"geometry\": {\"width\": %u, \"height\": %u, "
           "\"frames\": %u, \"fps\": 24},\n"
           "  \"seed\": %" PRIu64 ",\n"
           "  \"conditioning\": {\"images\": %u, \"first_image\": %s, "
           "\"last_image\": %s},\n"
           "  \"sampling\": {\"profile\": \"%s\", \"steps\": %u, "
           "\"turbo_adapter\": %s},\n"
           "  \"prompt_tokens\": %zu,\n"
           "  \"sequence_rows\": %zu,\n"
           "  \"metal\": {\"pipeline_archive\": \"%s\"},\n"
           "  \"durations_seconds\": {\"tokenizer\": %.6f, "
           "\"metal_setup\": %.6f, "
           "\"text_download\": %.6f, \"text_encode\": %.6f, "
           "\"image_encode\": %.6f, "
           "\"transformer_download\": %.6f, \"turbo_compile\": %.6f, "
           "\"rope_precompute\": %.6f, \"denoise\": %.6f, "
           "\"video_download\": %.6f, \"video_precompute\": %.6f, "
           "\"video_decode\": %.6f, "
           "\"audio_download\": %.6f, \"audio_decode\": %.6f, "
           "\"mux\": %.6f, \"total\": %.6f},\n"
           "  \"process_cpu_seconds\": {\"user\": %.6f, \"system\": %.6f},\n"
           "  \"peak_footprint_bytes\": %zu,\n"
           "  \"video_path\": ",
           options.width, options.height, options.frames, options.seed,
           result.condition_images,
           options.first_image_path != NULL ? "true" : "false",
           options.last_image_path != NULL ? "true" : "false",
           result.turbo_adapter_enabled ? "turbo-v4-step600-ema" : "base",
           result.sampling_steps,
           result.turbo_adapter_enabled ? "true" : "false",
           result.prompt_tokens, result.sequence_rows,
           result.pipeline_archive_hit ? "hit" : "miss",
           result.tokenizer_seconds, result.metal_setup_seconds,
           result.text_download_seconds,
           result.text_encode_seconds, result.image_encode_seconds,
           result.transformer_download_seconds,
           result.turbo_compile_seconds, result.rope_precompute_seconds,
           result.denoise_seconds,
           result.video_download_seconds,
           result.video_precompute_seconds,
           result.video_decode_seconds, result.audio_download_seconds,
           result.audio_decode_seconds, result.mux_seconds,
           result.total_seconds, result.process_user_seconds,
           result.process_system_seconds, result.peak_footprint_bytes);
    json_string(result.video_path);
    fputs(",\n  \"audio_path\": ", stdout);
    json_string(result.audio_path);
    fputs(",\n  \"output_path\": ", stdout);
    json_string(result.output_path);
    fputs("\n}\n", stdout);
    return 0;
}
