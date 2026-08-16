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
    int downstream_smoke = argc >= 2 &&
                           strcmp(argv[1], "--downstream-smoke") == 0;
    if ((downstream_smoke && argc != 3) ||
        (!downstream_smoke && (argc < 3 || argc > 7))) {
        fprintf(stderr,
                "usage: %s PROMPT OUTPUT_DIR [WIDTH HEIGHT FRAMES SEED]\n"
                "       %s --downstream-smoke OUTPUT_DIR\n",
                argv[0],
                argv[0]);
        return 2;
    }
    minimax_h3_m3_e2e_options options = {
        .width = !downstream_smoke && argc > 3 ?
            (uint32_t)strtoul(argv[3], NULL, 10) : 32u,
        .height = !downstream_smoke && argc > 4 ?
            (uint32_t)strtoul(argv[4], NULL, 10) : 32u,
        .frames = !downstream_smoke && argc > 5 ?
            (uint32_t)strtoul(argv[5], NULL, 10) : 22u,
        .seed = !downstream_smoke && argc > 6 ?
            strtoull(argv[6], NULL, 10) : UINT64_C(42),
        .prompt = downstream_smoke ? "[fixed conditioner state]" : argv[1],
        .tokenizer_image_path = "build/minimax-h3-tokenizer.h3tok",
        .metallib_path = "build/minimax-h3-m3-attention.metallib",
        .output_directory = argv[2],
    };
    minimax_h3_m3_e2e_result result;
    char error[1024];
    int status = downstream_smoke
        ? minimax_h3_m3_run_downstream_smoke(&options, &result, error,
                                             sizeof(error))
        : minimax_h3_m3_run_e2e(&options, &result, error, sizeof(error));
    if (status != 0) {
        fprintf(stderr, "MiniMax-H3 N-to-N failed: %s\n", error);
        return status;
    }
    printf("{\n  \"status\": \"%s\",\n  \"prompt\": ",
           downstream_smoke ? "downstream-smoke-passed" : "passed");
    json_string(options.prompt);
    printf(",\n  \"geometry\": {\"width\": %u, \"height\": %u, "
           "\"frames\": %u, \"fps\": 24},\n"
           "  \"seed\": %" PRIu64 ",\n"
           "  \"sampling\": {\"profile\": \"%s\", \"steps\": %u, "
           "\"turbo_adapter\": %s},\n"
           "  \"prompt_tokens\": %zu,\n"
           "  \"sequence_rows\": %zu,\n"
           "  \"durations_seconds\": {\"tokenizer\": %.6f, "
           "\"text_download\": %.6f, \"text_encode\": %.6f, "
           "\"transformer_download\": %.6f, \"turbo_compile\": %.6f, "
           "\"denoise\": %.6f, "
           "\"video_download\": %.6f, \"video_decode\": %.6f, "
           "\"audio_download\": %.6f, \"audio_decode\": %.6f, "
           "\"mux\": %.6f, \"total\": %.6f},\n"
           "  \"process_cpu_seconds\": {\"user\": %.6f, \"system\": %.6f},\n"
           "  \"peak_footprint_bytes\": %zu,\n"
           "  \"video_path\": ",
           options.width, options.height, options.frames, options.seed,
           result.turbo_adapter_enabled ? "turbo-v4-step600-ema" : "base",
           result.sampling_steps,
           result.turbo_adapter_enabled ? "true" : "false",
           result.prompt_tokens, result.sequence_rows,
           result.tokenizer_seconds, result.text_download_seconds,
           result.text_encode_seconds, result.transformer_download_seconds,
           result.turbo_compile_seconds, result.denoise_seconds,
           result.video_download_seconds,
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
