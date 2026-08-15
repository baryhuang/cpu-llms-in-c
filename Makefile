CC ?= cc

CPPFLAGS ?=
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS ?= -pthread
OMPFLAGS ?=

BUILD_DIR := build
TARGET_PROBE := $(BUILD_DIR)/target-probe
GEMMA4_LAYER_TEST := $(BUILD_DIR)/gemma4-layer-test
GEMMA4_TASK := $(BUILD_DIR)/gemma4-task
GEMMA4_LAYER_FIXTURE := tests/fixtures/gemma4_layer_v1.bin
QWEN35_GENERIC := models/qwen3.5-0.8b/targets/generic
QWEN35_LAYER_TEST := $(BUILD_DIR)/qwen35-layer-test
QWEN35_LAYER_FIXTURE := tests/fixtures/qwen35_layer_v1.bin
QWEN35_A113X := models/qwen3.5-0.8b/targets/a113x
QWEN35_TASK := $(BUILD_DIR)/qwen35-task
QWEN35_A113X_TASK := $(BUILD_DIR)/qwen35-task-a113x
WHISPER_SMALL_GENERIC := models/whisper-small.en/targets/generic
WHISPER_SMALL_LOG_MEL_TEST := $(BUILD_DIR)/whisper-small-log-mel-test
WHISPER_SMALL_LOG_MEL_FIXTURE := tests/fixtures/whisper_log_mel_80_v1.bin
WHISPER_ENCODER_STEM_TEST := $(BUILD_DIR)/whisper-encoder-stem-test
WHISPER_ENCODER_STEM_FIXTURE := tests/fixtures/whisper_encoder_stem_v1.bin
WHISPER_ENCODER_BLOCK_TEST := $(BUILD_DIR)/whisper-encoder-block-test
WHISPER_ENCODER_BLOCK_FIXTURE := tests/fixtures/whisper_encoder_block_v1.bin
WHISPER_SMALL_ENCODER_CHECK := $(BUILD_DIR)/whisper-small-encoder-check
WHISPER_SMALL_ENCODER_BENCH := $(BUILD_DIR)/whisper-small-encoder-bench
WHISPER_SMALL_A113X := models/whisper-small.en/targets/a113x
WHISPER_SMALL_A113X_BENCH := $(BUILD_DIR)/whisper-small-encoder-bench-a113x
WHISPER_SMALL_A113X_CHECK := $(BUILD_DIR)/whisper-small-encoder-check-a113x
WHISPER_SMALL_DECODER_CHECK := $(BUILD_DIR)/whisper-small-decoder-check
WHISPER_SMALL_A113X_DECODER_CHECK := $(BUILD_DIR)/whisper-small-decoder-check-a113x
WHISPER_SMALL_TRANSCRIBE := $(BUILD_DIR)/whisper-small-transcribe
WHISPER_SMALL_A113X_TRANSCRIBE := $(BUILD_DIR)/whisper-small-transcribe-a113x
QWEN36_M3 := models/qwen3.6-27b/targets/apple-m3-pro
QWEN36_M3_AIR := $(BUILD_DIR)/qwen36-m3-q4.air
QWEN36_M3_DELTANET_AIR := $(BUILD_DIR)/qwen36-m3-deltanet.air
QWEN36_M3_LAYER_AIR := $(BUILD_DIR)/qwen36-m3-layer.air
QWEN36_M3_ATTENTION_AIR := $(BUILD_DIR)/qwen36-m3-attention.air
QWEN36_M3_GLOBAL_AIR := $(BUILD_DIR)/qwen36-m3-global.air
QWEN36_M3_PREFILL_AIR := $(BUILD_DIR)/qwen36-m3-prefill.air
QWEN36_M3_METALLIB := $(BUILD_DIR)/qwen36-m3-q4.metallib
QWEN36_M3_RUNTIME_OBJECT := $(BUILD_DIR)/qwen36-m3.o
QWEN36_M3_BENCH_OBJECT := $(BUILD_DIR)/qwen36-m3-mlp-bench.o
QWEN36_M3_BENCH := $(BUILD_DIR)/qwen36-m3-mlp-bench
QWEN36_M3_DELTANET_OBJECT := $(BUILD_DIR)/qwen36-m3-deltanet.o
QWEN36_M3_DELTANET_BENCH_OBJECT := $(BUILD_DIR)/qwen36-m3-deltanet-bench.o
QWEN36_M3_DELTANET_BENCH := $(BUILD_DIR)/qwen36-m3-deltanet-bench
QWEN36_M3_LAYER_OBJECT := $(BUILD_DIR)/qwen36-m3-layer.o
QWEN36_M3_LAYER_BENCH_OBJECT := $(BUILD_DIR)/qwen36-m3-layer-bench.o
QWEN36_M3_LAYER_BENCH := $(BUILD_DIR)/qwen36-m3-layer-bench
QWEN36_M3_ATTENTION_OBJECT := $(BUILD_DIR)/qwen36-m3-attention.o
QWEN36_M3_ATTENTION_BENCH_OBJECT := $(BUILD_DIR)/qwen36-m3-attention-bench.o
QWEN36_M3_ATTENTION_BENCH := $(BUILD_DIR)/qwen36-m3-attention-bench
QWEN36_IMPORT := models/qwen3.6-27b/import
QWEN36_SAFETENSORS_INSPECT := $(BUILD_DIR)/qwen36-safetensors-inspect
QWEN36_SAFETENSORS_TEST := $(BUILD_DIR)/qwen36-safetensors-test
QWEN36_SHA256_TEST := $(BUILD_DIR)/qwen36-sha256-test
QWEN36_M3_PACK := $(BUILD_DIR)/qwen36-m3-pack
QWEN36_M3_ATTENTION_PACK := $(BUILD_DIR)/qwen36-m3-attention-pack
QWEN36_M3_GLOBAL_PACK := $(BUILD_DIR)/qwen36-m3-global-pack
QWEN36_M3_OMLX_EXPORT := $(BUILD_DIR)/qwen36-m3-export-omlx
QWEN36_MTP_PACK := $(BUILD_DIR)/qwen36-mtp-pack
QWEN38_MTP_PACK := $(BUILD_DIR)/qwen38-mtp-pack
QWEN36_M3_DECODE_OBJECT := $(BUILD_DIR)/qwen36-m3-decode.o
QWEN36_M3_DECODE_CLI_OBJECT := $(BUILD_DIR)/qwen36-m3-decode-cli.o
QWEN36_M3_DECODE := $(BUILD_DIR)/qwen36-m3-decode
QWEN36_TOKENIZER_PACK := $(BUILD_DIR)/qwen36-tokenizer-pack
QWEN36_TOKENIZER_OBJECT := $(BUILD_DIR)/qwen36-tokenizer.o
QWEN36_TOKENIZER_CLI_OBJECT := $(BUILD_DIR)/qwen36-tokenizer-cli.o
QWEN36_TOKENIZER_CLI := $(BUILD_DIR)/qwen36-tokenizer
QWEN36_SAMPLER_OBJECT := $(BUILD_DIR)/qwen36-sampler.o
QWEN36_M3_GENERATE_OBJECT := $(BUILD_DIR)/qwen36-m3-generate-cli.o
QWEN36_M3_GENERATE := $(BUILD_DIR)/qwen36-m3-generate
QWEN36_M3_CHAT_OBJECT := $(BUILD_DIR)/qwen36-m3-chat-cli.o
QWEN36_M3_CHAT := $(BUILD_DIR)/qwen36-m3-chat
QWEN36_SAMPLER_TEST := $(BUILD_DIR)/qwen36-sampler-test
QWEN36_M3_API_STATE_TEST := $(BUILD_DIR)/qwen36-m3-api-state-test
QWEN36_M3_PREFILL_PARITY_TEST := $(BUILD_DIR)/qwen36-m3-prefill-parity-test

.PHONY: all a113x clean fixture linux-tools qwen36-m3-bench qwen36-m3-deltanet-bench qwen36-m3-layer-bench qwen36-m3-attention-bench qwen36-m3-decode qwen36-m3-generate qwen36-m3-chat qwen36-m3-api-state-test qwen36-m3-prefill-parity-test qwen36-tools test whisper-small-tools

all: $(GEMMA4_LAYER_TEST) $(GEMMA4_TASK)

linux-tools: $(TARGET_PROBE)

a113x: $(QWEN35_A113X_TASK) $(WHISPER_SMALL_A113X_BENCH) \
	$(WHISPER_SMALL_A113X_CHECK) $(WHISPER_SMALL_A113X_TRANSCRIBE) \
	$(WHISPER_SMALL_A113X_DECODER_CHECK)

whisper-small-tools: $(WHISPER_SMALL_ENCODER_CHECK) $(WHISPER_SMALL_ENCODER_BENCH) \
	$(WHISPER_SMALL_DECODER_CHECK) $(WHISPER_SMALL_TRANSCRIBE)

qwen36-m3-bench: $(QWEN36_M3_BENCH) $(QWEN36_M3_METALLIB)
	$(QWEN36_M3_BENCH) $(QWEN36_M3_METALLIB)

qwen36-m3-deltanet-bench: $(QWEN36_M3_DELTANET_BENCH) $(QWEN36_M3_METALLIB)
	$(QWEN36_M3_DELTANET_BENCH) $(QWEN36_M3_METALLIB)

qwen36-m3-layer-bench: $(QWEN36_M3_LAYER_BENCH) $(QWEN36_M3_METALLIB)

qwen36-m3-attention-bench: $(QWEN36_M3_ATTENTION_BENCH) \
	$(QWEN36_M3_METALLIB)

qwen36-m3-decode: $(QWEN36_M3_DECODE) $(QWEN36_M3_METALLIB)

qwen36-m3-generate: $(QWEN36_M3_GENERATE) $(QWEN36_M3_METALLIB)

qwen36-m3-chat: $(QWEN36_M3_CHAT) $(QWEN36_M3_METALLIB)

# Needs the packed model directory and metallib, so it is not part of the
# fixture-only `test` target. Run it manually:
#   build/qwen36-m3-api-state-test <model-directory> <metallib>
qwen36-m3-api-state-test: $(QWEN36_M3_API_STATE_TEST) $(QWEN36_M3_METALLIB)

# Live bitwise parity between batched prefill and one-token decode. Run:
#   build/qwen36-m3-prefill-parity-test <model-directory> <metallib>
qwen36-m3-prefill-parity-test: $(QWEN36_M3_PREFILL_PARITY_TEST) \
	$(QWEN36_M3_METALLIB)

qwen36-mtp-pack: $(QWEN36_MTP_PACK)

qwen38-mtp-pack: $(QWEN38_MTP_PACK)

qwen36-tools: $(QWEN36_SAFETENSORS_INSPECT) $(QWEN36_M3_PACK) \
	$(QWEN36_M3_ATTENTION_PACK) $(QWEN36_M3_GLOBAL_PACK) \
	$(QWEN36_M3_OMLX_EXPORT) $(QWEN36_TOKENIZER_PACK) \
	$(QWEN36_TOKENIZER_CLI)

fixture: $(GEMMA4_LAYER_FIXTURE)

test: $(GEMMA4_LAYER_TEST) $(GEMMA4_LAYER_FIXTURE) $(QWEN35_LAYER_TEST) $(QWEN35_LAYER_FIXTURE) \
	$(WHISPER_SMALL_LOG_MEL_TEST) $(WHISPER_SMALL_LOG_MEL_FIXTURE) \
	$(WHISPER_ENCODER_STEM_TEST) $(WHISPER_ENCODER_STEM_FIXTURE) \
	$(WHISPER_ENCODER_BLOCK_TEST) $(WHISPER_ENCODER_BLOCK_FIXTURE) \
	$(QWEN36_SAFETENSORS_TEST) $(QWEN36_SHA256_TEST) \
	$(QWEN36_SAMPLER_TEST)
	python3 -m unittest discover -s tests -p 'test_*.py'
	$(GEMMA4_LAYER_TEST) $(GEMMA4_LAYER_FIXTURE)
	$(QWEN35_LAYER_TEST) $(QWEN35_LAYER_FIXTURE)
	$(WHISPER_SMALL_LOG_MEL_TEST) $(WHISPER_SMALL_LOG_MEL_FIXTURE)
	$(WHISPER_ENCODER_STEM_TEST) $(WHISPER_ENCODER_STEM_FIXTURE)
	$(WHISPER_ENCODER_BLOCK_TEST) $(WHISPER_ENCODER_BLOCK_FIXTURE)
	$(QWEN36_SAFETENSORS_TEST)
	$(QWEN36_SHA256_TEST)
	$(QWEN36_SAMPLER_TEST)

$(TARGET_PROBE): tools/target_probe.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

GEMMA4_GENERIC := models/gemma-4-e2b/targets/generic

$(GEMMA4_LAYER_TEST): tests/gemma4_layer_test.c $(GEMMA4_GENERIC)/gemma4_layer.c $(GEMMA4_GENERIC)/gemma4_layer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(GEMMA4_GENERIC) tests/gemma4_layer_test.c $(GEMMA4_GENERIC)/gemma4_layer.c \
		-o $@ $(LDFLAGS) -lm

$(GEMMA4_TASK): $(GEMMA4_GENERIC)/gemma4_task.c $(GEMMA4_GENERIC)/gemma4_task.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -I$(GEMMA4_GENERIC) $(GEMMA4_GENERIC)/gemma4_task.c \
		-o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(GEMMA4_LAYER_FIXTURE): compiler/generate_gemma4_layer_fixture.py
	python3 $< --output $@

$(QWEN35_LAYER_TEST): tests/qwen35_layer_test.c $(QWEN35_GENERIC)/qwen35_layer.c $(QWEN35_GENERIC)/qwen35_layer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN35_GENERIC) tests/qwen35_layer_test.c $(QWEN35_GENERIC)/qwen35_layer.c \
		-o $@ $(LDFLAGS) -lm

$(QWEN35_LAYER_FIXTURE): compiler/generate_qwen35_layer_fixture.py
	python3 $< --output $@

$(QWEN35_TASK): $(QWEN35_GENERIC)/qwen35_task.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(QWEN35_A113X_TASK): $(QWEN35_A113X)/qwen35_task.c $(QWEN35_A113X)/qwen35_a113x_kernels.h $(QWEN35_GENERIC)/qwen35_task.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -mcpu=cortex-a53 -mtune=cortex-a53 $< \
		-o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(WHISPER_SMALL_LOG_MEL_TEST): tests/whisper_small_log_mel_test.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c $(WHISPER_SMALL_GENERIC)/whisper_small_frontend.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) tests/whisper_small_log_mel_test.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c -o $@ $(LDFLAGS) -lm

$(WHISPER_SMALL_LOG_MEL_FIXTURE): compiler/generate_whisper_log_mel_fixture.py
	python3 $< --n-mels 80 --output $@

$(WHISPER_ENCODER_STEM_TEST): tests/whisper_encoder_stem_test.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c $(WHISPER_SMALL_GENERIC)/whisper_small_frontend.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) tests/whisper_encoder_stem_test.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c -o $@ $(LDFLAGS) -lm

$(WHISPER_ENCODER_STEM_FIXTURE): compiler/generate_whisper_encoder_stem_fixture.py
	python3 $< --output $@

$(WHISPER_ENCODER_BLOCK_TEST): tests/whisper_encoder_block_test.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_encoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) tests/whisper_encoder_block_test.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c -o $@ $(LDFLAGS) -lm

$(WHISPER_ENCODER_BLOCK_FIXTURE): compiler/generate_whisper_encoder_block_fixture.py
	python3 $< --output $@

$(WHISPER_SMALL_ENCODER_CHECK): tools/whisper_small_encoder_check.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_encoder.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c $(WHISPER_SMALL_GENERIC)/whisper_small_frontend.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_decoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) tools/whisper_small_encoder_check.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c -o $@ $(LDFLAGS) -lm

$(WHISPER_SMALL_ENCODER_BENCH): tools/whisper_small_encoder_bench.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_encoder.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c $(WHISPER_SMALL_GENERIC)/whisper_small_frontend.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_decoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) tools/whisper_small_encoder_bench.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c -o $@ $(LDFLAGS) -lm

$(WHISPER_SMALL_DECODER_CHECK): tools/whisper_small_decoder_check.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_encoder.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c $(WHISPER_SMALL_GENERIC)/whisper_small_frontend.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_decoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) tools/whisper_small_decoder_check.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c -o $@ $(LDFLAGS) -lm

$(WHISPER_SMALL_TRANSCRIBE): tools/whisper_small_transcribe.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_encoder.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c $(WHISPER_SMALL_GENERIC)/whisper_small_frontend.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_decoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) tools/whisper_small_transcribe.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c -o $@ $(LDFLAGS) -lm

$(WHISPER_SMALL_A113X_TRANSCRIBE): tools/whisper_small_transcribe.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_A113X)/whisper_small_encoder.c $(WHISPER_SMALL_A113X)/whisper_small_a113x_kernels.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
	$(WHISPER_SMALL_A113X)/whisper_small_decoder.c \
	$(WHISPER_SMALL_A113X)/whisper_small_a113x_decoder_kernels.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -mcpu=cortex-a53 -mtune=cortex-a53 \
		-I$(WHISPER_SMALL_GENERIC) tools/whisper_small_transcribe.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_A113X)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
		$(WHISPER_SMALL_A113X)/whisper_small_decoder.c -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(WHISPER_SMALL_A113X_DECODER_CHECK): tools/whisper_small_decoder_check.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
	$(WHISPER_SMALL_A113X)/whisper_small_decoder.c \
	$(WHISPER_SMALL_A113X)/whisper_small_a113x_decoder_kernels.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -mcpu=cortex-a53 -mtune=cortex-a53 \
		-I$(WHISPER_SMALL_GENERIC) tools/whisper_small_decoder_check.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
		$(WHISPER_SMALL_A113X)/whisper_small_decoder.c -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(WHISPER_SMALL_A113X_BENCH): tools/whisper_small_encoder_bench.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_A113X)/whisper_small_encoder.c $(WHISPER_SMALL_A113X)/whisper_small_a113x_kernels.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -mcpu=cortex-a53 -mtune=cortex-a53 \
		-I$(WHISPER_SMALL_GENERIC) tools/whisper_small_encoder_bench.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_A113X)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(WHISPER_SMALL_A113X_CHECK): tools/whisper_small_encoder_check.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_A113X)/whisper_small_encoder.c $(WHISPER_SMALL_A113X)/whisper_small_a113x_kernels.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -mcpu=cortex-a53 -mtune=cortex-a53 \
		-I$(WHISPER_SMALL_GENERIC) tools/whisper_small_encoder_check.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_A113X)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(QWEN36_M3_AIR): $(QWEN36_M3)/qwen36_q4.metal
	mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $@

$(QWEN36_M3_DELTANET_AIR): $(QWEN36_M3)/qwen36_deltanet.metal
	mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $@

$(QWEN36_M3_LAYER_AIR): $(QWEN36_M3)/qwen36_layer.metal
	mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $@

$(QWEN36_M3_ATTENTION_AIR): $(QWEN36_M3)/qwen36_attention.metal
	mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $@

$(QWEN36_M3_GLOBAL_AIR): $(QWEN36_M3)/qwen36_global.metal
	mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $@

$(QWEN36_M3_PREFILL_AIR): $(QWEN36_M3)/qwen36_prefill.metal
	mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $@

$(QWEN36_M3_METALLIB): $(QWEN36_M3_AIR) $(QWEN36_M3_DELTANET_AIR) \
	$(QWEN36_M3_LAYER_AIR) $(QWEN36_M3_ATTENTION_AIR) \
	$(QWEN36_M3_GLOBAL_AIR) $(QWEN36_M3_PREFILL_AIR)
	xcrun -sdk macosx metallib $^ -o $@

$(QWEN36_M3_RUNTIME_OBJECT): $(QWEN36_M3)/qwen36_m3.m $(QWEN36_M3)/qwen36_m3.h \
	$(QWEN36_M3)/qwen36_m3_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -fobjc-arc -I$(QWEN36_M3) -c $< -o $@

$(QWEN36_M3_BENCH_OBJECT): tools/qwen36_m3_mlp_bench.c $(QWEN36_M3)/qwen36_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) -c $< -o $@

$(QWEN36_M3_BENCH): $(QWEN36_M3_RUNTIME_OBJECT) $(QWEN36_M3_BENCH_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal -lm

$(QWEN36_M3_DELTANET_OBJECT): $(QWEN36_M3)/qwen36_m3_deltanet.m \
	$(QWEN36_M3)/qwen36_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -fobjc-arc -I$(QWEN36_M3) -c $< -o $@

$(QWEN36_M3_DELTANET_BENCH_OBJECT): tools/qwen36_m3_deltanet_bench.c \
	$(QWEN36_M3)/qwen36_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) -c $< -o $@

$(QWEN36_M3_DELTANET_BENCH): $(QWEN36_M3_DELTANET_OBJECT) \
	$(QWEN36_M3_DELTANET_BENCH_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal -lm

$(QWEN36_M3_LAYER_OBJECT): $(QWEN36_M3)/qwen36_m3_layer.m \
	$(QWEN36_M3)/qwen36_m3.h $(QWEN36_M3)/qwen36_m3_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -fobjc-arc \
		-I$(QWEN36_M3) -c $< -o $@

$(QWEN36_M3_LAYER_BENCH_OBJECT): tools/qwen36_m3_layer_bench.c \
	$(QWEN36_M3)/qwen36_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) -c $< -o $@

$(QWEN36_M3_LAYER_BENCH): $(QWEN36_M3_LAYER_OBJECT) \
	$(QWEN36_M3_LAYER_BENCH_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal -lm

$(QWEN36_M3_ATTENTION_OBJECT): $(QWEN36_M3)/qwen36_m3_attention.m \
	$(QWEN36_M3)/qwen36_m3.h $(QWEN36_M3)/qwen36_m3_image.h \
	$(QWEN36_M3)/qwen36_m3_attention_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -fobjc-arc \
		-I$(QWEN36_M3) -c $< -o $@

$(QWEN36_M3_ATTENTION_BENCH_OBJECT): tools/qwen36_m3_attention_bench.c \
	$(QWEN36_M3)/qwen36_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) -c $< -o $@

$(QWEN36_M3_ATTENTION_BENCH): $(QWEN36_M3_ATTENTION_OBJECT) \
	$(QWEN36_M3_ATTENTION_BENCH_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal -lm

$(QWEN36_M3_DECODE_OBJECT): $(QWEN36_M3)/qwen36_m3_decode.m \
	$(QWEN36_M3)/qwen36_m3_decode.h $(QWEN36_M3)/qwen36_m3.h \
	$(QWEN36_M3)/qwen36_m3_image.h \
	$(QWEN36_M3)/qwen36_m3_attention_image.h \
	$(QWEN36_M3)/qwen36_m3_global_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -fobjc-arc \
		-I$(QWEN36_M3) -c $< -o $@

$(QWEN36_M3_DECODE_CLI_OBJECT): tools/qwen36_m3_decode.c \
	$(QWEN36_M3)/qwen36_m3_decode.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) -c $< -o $@

$(QWEN36_M3_DECODE): $(QWEN36_M3_DECODE_OBJECT) \
	$(QWEN36_M3_DECODE_CLI_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal -lm

$(QWEN36_TOKENIZER_OBJECT): $(QWEN36_M3)/qwen36_tokenizer.c \
	$(QWEN36_M3)/qwen36_tokenizer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) -c $< -o $@

$(QWEN36_TOKENIZER_CLI_OBJECT): tools/qwen36_tokenizer.c \
	$(QWEN36_M3)/qwen36_tokenizer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) -c $< -o $@

$(QWEN36_TOKENIZER_CLI): $(QWEN36_TOKENIZER_OBJECT) \
	$(QWEN36_TOKENIZER_CLI_OBJECT)
	$(CC) $^ -o $@ -framework CoreFoundation -licucore

$(QWEN36_SAMPLER_OBJECT): $(QWEN36_M3)/qwen36_sampler.c \
	$(QWEN36_M3)/qwen36_sampler.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) -c $< -o $@

$(QWEN36_SAMPLER_TEST): tests/qwen36_sampler_test.c \
	$(QWEN36_M3)/qwen36_sampler.c $(QWEN36_M3)/qwen36_sampler.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) \
		tests/qwen36_sampler_test.c $(QWEN36_M3)/qwen36_sampler.c \
		-o $@ -lm

$(QWEN36_M3_GENERATE_OBJECT): tools/qwen36_m3_generate.c \
	$(QWEN36_M3)/qwen36_m3_decode.h $(QWEN36_M3)/qwen36_tokenizer.h \
	$(QWEN36_M3)/qwen36_sampler.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) -c $< -o $@

$(QWEN36_M3_GENERATE): $(QWEN36_M3_DECODE_OBJECT) \
	$(QWEN36_TOKENIZER_OBJECT) $(QWEN36_SAMPLER_OBJECT) \
	$(QWEN36_M3_GENERATE_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal \
		-framework CoreFoundation -licucore -lm

$(QWEN36_M3_CHAT_OBJECT): tools/qwen36_m3_chat.c \
	$(QWEN36_M3)/qwen36_m3_decode.h $(QWEN36_M3)/qwen36_tokenizer.h \
	$(QWEN36_M3)/qwen36_sampler.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) -c $< -o $@

$(QWEN36_M3_CHAT): $(QWEN36_M3_DECODE_OBJECT) \
	$(QWEN36_TOKENIZER_OBJECT) $(QWEN36_SAMPLER_OBJECT) \
	$(QWEN36_M3_CHAT_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal \
		-framework CoreFoundation -licucore -lm

$(QWEN36_M3_API_STATE_TEST): tests/qwen36_m3_api_state_test.c \
	$(QWEN36_M3_DECODE_OBJECT) $(QWEN36_M3)/qwen36_m3_decode.h \
	$(QWEN36_M3)/qwen36_m3_global_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) \
		tests/qwen36_m3_api_state_test.c $(QWEN36_M3_DECODE_OBJECT) \
		-o $@ -framework Foundation -framework Metal -lm

$(QWEN36_M3_PREFILL_PARITY_TEST): tests/qwen36_m3_prefill_parity_test.c \
	$(QWEN36_M3_DECODE_OBJECT) $(QWEN36_M3)/qwen36_m3_decode.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) \
		tests/qwen36_m3_prefill_parity_test.c \
		$(QWEN36_M3_DECODE_OBJECT) \
		-o $@ -framework Foundation -framework Metal -lm

$(QWEN36_SAFETENSORS_INSPECT): tools/qwen36_safetensors_inspect.c \
	$(QWEN36_IMPORT)/qwen36_safetensors.c $(QWEN36_IMPORT)/qwen36_safetensors.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_IMPORT) tools/qwen36_safetensors_inspect.c \
		$(QWEN36_IMPORT)/qwen36_safetensors.c -o $@

$(QWEN36_SAFETENSORS_TEST): tests/qwen36_safetensors_test.c \
	$(QWEN36_IMPORT)/qwen36_safetensors.c $(QWEN36_IMPORT)/qwen36_safetensors.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_IMPORT) tests/qwen36_safetensors_test.c \
		$(QWEN36_IMPORT)/qwen36_safetensors.c -o $@

$(QWEN36_SHA256_TEST): tests/qwen36_sha256_test.c \
	$(QWEN36_IMPORT)/qwen36_sha256.c $(QWEN36_IMPORT)/qwen36_sha256.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_IMPORT) tests/qwen36_sha256_test.c \
		$(QWEN36_IMPORT)/qwen36_sha256.c -o $@

$(QWEN36_M3_PACK): tools/qwen36_m3_pack.c \
	$(QWEN36_IMPORT)/qwen36_sha256.c $(QWEN36_IMPORT)/qwen36_sha256.h \
	$(QWEN36_IMPORT)/qwen36_safetensors.c $(QWEN36_IMPORT)/qwen36_safetensors.h \
	$(QWEN36_M3)/qwen36_m3.h $(QWEN36_M3)/qwen36_m3_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_IMPORT) -I$(QWEN36_M3) \
		tools/qwen36_m3_pack.c $(QWEN36_IMPORT)/qwen36_sha256.c \
		$(QWEN36_IMPORT)/qwen36_safetensors.c -o $@ -lm

$(QWEN36_MTP_PACK): tools/qwen36_mtp_pack.c \
	$(QWEN36_IMPORT)/qwen36_sha256.c $(QWEN36_IMPORT)/qwen36_sha256.h \
	$(QWEN36_IMPORT)/qwen36_safetensors.c $(QWEN36_IMPORT)/qwen36_safetensors.h \
	$(QWEN36_M3)/qwen36_m3.h $(QWEN36_M3)/qwen36_m3_image.h \
	$(QWEN36_M3)/qwen36_m3_attention_image.h \
	$(QWEN36_M3)/qwen36_m3_mtp_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_IMPORT) -I$(QWEN36_M3) \
		tools/qwen36_mtp_pack.c \
		$(QWEN36_IMPORT)/qwen36_sha256.c \
		$(QWEN36_IMPORT)/qwen36_safetensors.c -o $@ -lm

$(QWEN38_MTP_PACK): tools/qwen38_mtp_pack.c \
	$(QWEN36_IMPORT)/qwen36_sha256.c $(QWEN36_IMPORT)/qwen36_sha256.h \
	$(QWEN36_IMPORT)/qwen36_safetensors.c $(QWEN36_IMPORT)/qwen36_safetensors.h \
	$(QWEN36_M3)/qwen36_m3.h $(QWEN36_M3)/qwen36_m3_image.h \
	$(QWEN36_M3)/qwen36_m3_attention_image.h \
	$(QWEN36_M3)/qwen36_m3_mtp_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_IMPORT) -I$(QWEN36_M3) \
		tools/qwen38_mtp_pack.c \
		$(QWEN36_IMPORT)/qwen36_sha256.c \
		$(QWEN36_IMPORT)/qwen36_safetensors.c -o $@ -lm

$(QWEN36_M3_ATTENTION_PACK): tools/qwen36_m3_attention_pack.c \
	$(QWEN36_IMPORT)/qwen36_sha256.c $(QWEN36_IMPORT)/qwen36_sha256.h \
	$(QWEN36_IMPORT)/qwen36_safetensors.c $(QWEN36_IMPORT)/qwen36_safetensors.h \
	$(QWEN36_M3)/qwen36_m3.h $(QWEN36_M3)/qwen36_m3_image.h \
	$(QWEN36_M3)/qwen36_m3_attention_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_IMPORT) -I$(QWEN36_M3) \
		tools/qwen36_m3_attention_pack.c $(QWEN36_IMPORT)/qwen36_sha256.c \
		$(QWEN36_IMPORT)/qwen36_safetensors.c -o $@

$(QWEN36_M3_GLOBAL_PACK): tools/qwen36_m3_global_pack.c \
	$(QWEN36_IMPORT)/qwen36_sha256.c $(QWEN36_IMPORT)/qwen36_sha256.h \
	$(QWEN36_IMPORT)/qwen36_safetensors.c $(QWEN36_IMPORT)/qwen36_safetensors.h \
	$(QWEN36_M3)/qwen36_m3.h $(QWEN36_M3)/qwen36_m3_image.h \
	$(QWEN36_M3)/qwen36_m3_global_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_IMPORT) -I$(QWEN36_M3) \
		tools/qwen36_m3_global_pack.c $(QWEN36_IMPORT)/qwen36_sha256.c \
		$(QWEN36_IMPORT)/qwen36_safetensors.c -o $@

$(QWEN36_M3_OMLX_EXPORT): tools/qwen36_m3_export_omlx.c \
	$(QWEN36_M3)/qwen36_m3_image.h \
	$(QWEN36_M3)/qwen36_m3_attention_image.h \
	$(QWEN36_M3)/qwen36_m3_global_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_M3) $< -o $@

$(QWEN36_TOKENIZER_PACK): tools/qwen36_tokenizer_pack.c \
	$(QWEN36_M3)/qwen36_tokenizer.h \
	$(QWEN36_IMPORT)/qwen36_sha256.c $(QWEN36_IMPORT)/qwen36_sha256.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN36_IMPORT) -I$(QWEN36_M3) \
		tools/qwen36_tokenizer_pack.c $(QWEN36_IMPORT)/qwen36_sha256.c \
		-o $@

clean:
	rm -f $(TARGET_PROBE) $(GEMMA4_LAYER_TEST) $(GEMMA4_TASK) $(QWEN35_LAYER_TEST) \
		$(QWEN35_TASK) $(QWEN35_A113X_TASK) \
		$(WHISPER_SMALL_LOG_MEL_TEST) $(WHISPER_ENCODER_STEM_TEST) \
		$(WHISPER_ENCODER_BLOCK_TEST) $(WHISPER_SMALL_ENCODER_CHECK) \
		$(WHISPER_SMALL_ENCODER_BENCH) $(WHISPER_SMALL_A113X_BENCH) \
		$(WHISPER_SMALL_A113X_CHECK) $(WHISPER_SMALL_DECODER_CHECK) \
		$(WHISPER_SMALL_A113X_DECODER_CHECK) $(WHISPER_SMALL_TRANSCRIBE) \
		$(WHISPER_SMALL_A113X_TRANSCRIBE) $(QWEN36_M3_AIR) \
		$(QWEN36_M3_DELTANET_AIR) $(QWEN36_M3_METALLIB) \
		$(QWEN36_M3_LAYER_AIR) \
		$(QWEN36_M3_ATTENTION_AIR) \
		$(QWEN36_M3_GLOBAL_AIR) \
		$(QWEN36_M3_RUNTIME_OBJECT) $(QWEN36_M3_BENCH_OBJECT) $(QWEN36_M3_BENCH) \
		$(QWEN36_M3_DELTANET_OBJECT) $(QWEN36_M3_DELTANET_BENCH_OBJECT) \
		$(QWEN36_M3_DELTANET_BENCH) $(QWEN36_M3_LAYER_OBJECT) \
		$(QWEN36_M3_LAYER_BENCH_OBJECT) $(QWEN36_M3_LAYER_BENCH)
	rm -f $(QWEN36_M3_ATTENTION_OBJECT) $(QWEN36_M3_ATTENTION_BENCH_OBJECT) \
		$(QWEN36_M3_ATTENTION_BENCH)
	rm -f $(QWEN36_M3_DECODE_OBJECT) $(QWEN36_M3_DECODE_CLI_OBJECT) \
		$(QWEN36_M3_DECODE) $(QWEN36_M3_GENERATE_OBJECT) \
		$(QWEN36_M3_GENERATE)
	rm -f $(QWEN36_SAFETENSORS_INSPECT) $(QWEN36_SAFETENSORS_TEST) \
		$(QWEN36_SHA256_TEST) $(QWEN36_M3_PACK) \
		$(QWEN36_M3_ATTENTION_PACK) $(QWEN36_M3_GLOBAL_PACK) \
		$(QWEN36_M3_OMLX_EXPORT) \
		$(QWEN36_TOKENIZER_PACK) $(QWEN36_TOKENIZER_OBJECT) \
		$(QWEN36_TOKENIZER_CLI_OBJECT) $(QWEN36_TOKENIZER_CLI) \
		$(QWEN36_SAMPLER_OBJECT) $(QWEN36_SAMPLER_TEST)
