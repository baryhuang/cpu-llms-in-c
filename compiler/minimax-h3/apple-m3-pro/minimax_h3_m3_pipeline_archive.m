#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>

static void print_json_string(const char *value) {
    putchar('"');
    for (const unsigned char *cursor = (const unsigned char *)value;
         *cursor != '\0'; ++cursor) {
        if (*cursor == '"' || *cursor == '\\') putchar('\\');
        if (*cursor >= 0x20u) putchar(*cursor);
    }
    putchar('"');
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s INPUT.metallib OUTPUT.mtlarchive\n", argv[0]);
        return 2;
    }
    @autoreleasepool {
        static const char *names[] = {
            "minimax_h3_q4_gemm_bf16_meta",
            "minimax_h3_q4_gemm_bf16_activation",
            "minimax_h3_q8_gemm_bf16_meta",
            "minimax_h3_dense_bf16",
            "minimax_h3_dense_bf16_f16_to_bf16",
            "minimax_h3_dense_bf16_activation",
            "minimax_h3_dense_bf16_activation_add",
            "minimax_h3_dense_f32",
            "minimax_h3_dense_f32_f16_to_bf16",
            "minimax_h3_dense_f32_f32_to_bf16",
            "minimax_h3_dense_f32_bf16_to_f32",
            "minimax_h3_dense_f32_bf16_activation",
            "minimax_h3_dense_f16",
            "minimax_h3_dense_f16_mma",
            "minimax_h3_dense_f16_mma_weight_tiled",
            "minimax_h3_dense_f16_mma_weight_tiled_b64",
            "minimax_h3_rms_plain",
            "minimax_h3_rms_adaln",
            "minimax_h3_rms_plain_bf16",
            "minimax_h3_rms_adaln_bf16",
            "minimax_h3_rms_f16",
            "minimax_h3_layernorm_f16",
            "minimax_h3_scaled_residual_f16",
            "minimax_h3_silu_pair",
            "minimax_h3_silu_split",
            "minimax_h3_silu_split_bf16",
            "minimax_h3_gated_residual",
            "minimax_h3_gated_residual_bf16",
            "minimax_h3_qwen_prepare_qkv",
            "minimax_h3_qwen_causal_attention",
            "minimax_h3_build_rope_f32",
            "minimax_h3_prepare_qkv",
            "minimax_h3_hierarchical_attention",
            "minimax_h3_prepare_qkv_bf16",
            "minimax_h3_dense_attention_bf16",
            "minimax_h3_dense_attention_mma64_bf16",
            "minimax_h3_f32_to_bf16",
            "minimax_h3_build_video_rope_f32",
            "minimax_h3_video_prepare_qkv",
            "minimax_h3_video_attention",
            "minimax_h3_video_attention_tiled8",
            "minimax_h3_video_attention_tiled16",
            "minimax_h3_audio_conv1d_f32",
            "minimax_h3_audio_conv_transpose1d_f32",
            "minimax_h3_audio_alias_snake_f32",
            "minimax_h3_audio_residual_f32",
            "minimax_h3_audio_average3_f32",
        };
        NSError *error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSURL *library_url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library =
            [device newLibraryWithURL:library_url error:&error];
        MTLBinaryArchiveDescriptor *archive_descriptor =
            [MTLBinaryArchiveDescriptor new];
        id<MTLBinaryArchive> archive = device != nil
            ? [device newBinaryArchiveWithDescriptor:archive_descriptor
                                               error:&error]
            : nil;
        if (device == nil || library == nil || archive == nil) {
            const char *description = error.localizedDescription.UTF8String;
            fprintf(stderr, "pipeline archive setup failed: %s\n",
                    description != NULL ? description : "unknown error");
            return 1;
        }
        size_t count = sizeof(names) / sizeof(names[0]);
        for (size_t index = 0u; index < count; ++index) {
            NSString *name = [NSString stringWithUTF8String:names[index]];
            id<MTLFunction> function = [library newFunctionWithName:name];
            MTLComputePipelineDescriptor *pipeline =
                [MTLComputePipelineDescriptor new];
            pipeline.label = name;
            pipeline.computeFunction = function;
            if (function == nil ||
                ![archive addComputePipelineFunctionsWithDescriptor:pipeline
                                                               error:&error]) {
                const char *description = error.localizedDescription.UTF8String;
                fprintf(stderr, "cannot archive %s: %s\n", names[index],
                        description != NULL ? description : "missing function");
                return 1;
            }
        }
        NSURL *output_url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:argv[2]]];
        [[NSFileManager defaultManager] removeItemAtURL:output_url error:nil];
        if (![archive serializeToURL:output_url error:&error]) {
            const char *description = error.localizedDescription.UTF8String;
            fprintf(stderr, "cannot write pipeline archive: %s\n",
                    description != NULL ? description : "unknown error");
            return 1;
        }
        printf("{\n  \"device\": ");
        print_json_string(device.name.UTF8String);
        printf(",\n  \"pipeline_count\": %zu,\n  \"archive\": ", count);
        print_json_string(argv[2]);
        printf("\n}\n");
        return 0;
    }
}
