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

.PHONY: all a113x clean fixture linux-tools test whisper-small-tools

all: $(GEMMA4_LAYER_TEST) $(GEMMA4_TASK)

linux-tools: $(TARGET_PROBE)

a113x: $(QWEN35_A113X_TASK) $(WHISPER_SMALL_A113X_BENCH) \
	$(WHISPER_SMALL_A113X_CHECK) $(WHISPER_SMALL_A113X_TRANSCRIBE) \
	$(WHISPER_SMALL_A113X_DECODER_CHECK)

whisper-small-tools: $(WHISPER_SMALL_ENCODER_CHECK) $(WHISPER_SMALL_ENCODER_BENCH) \
	$(WHISPER_SMALL_DECODER_CHECK) $(WHISPER_SMALL_TRANSCRIBE)

fixture: $(GEMMA4_LAYER_FIXTURE)

test: $(GEMMA4_LAYER_TEST) $(GEMMA4_LAYER_FIXTURE) $(QWEN35_LAYER_TEST) $(QWEN35_LAYER_FIXTURE) \
	$(WHISPER_SMALL_LOG_MEL_TEST) $(WHISPER_SMALL_LOG_MEL_FIXTURE) \
	$(WHISPER_ENCODER_STEM_TEST) $(WHISPER_ENCODER_STEM_FIXTURE) \
	$(WHISPER_ENCODER_BLOCK_TEST) $(WHISPER_ENCODER_BLOCK_FIXTURE)
	python3 -m unittest discover -s tests -p 'test_*.py'
	$(GEMMA4_LAYER_TEST) $(GEMMA4_LAYER_FIXTURE)
	$(QWEN35_LAYER_TEST) $(QWEN35_LAYER_FIXTURE)
	$(WHISPER_SMALL_LOG_MEL_TEST) $(WHISPER_SMALL_LOG_MEL_FIXTURE)
	$(WHISPER_ENCODER_STEM_TEST) $(WHISPER_ENCODER_STEM_FIXTURE)
	$(WHISPER_ENCODER_BLOCK_TEST) $(WHISPER_ENCODER_BLOCK_FIXTURE)

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

clean:
	rm -f $(TARGET_PROBE) $(GEMMA4_LAYER_TEST) $(GEMMA4_TASK) $(QWEN35_LAYER_TEST) \
		$(QWEN35_TASK) $(QWEN35_A113X_TASK) \
		$(WHISPER_SMALL_LOG_MEL_TEST) $(WHISPER_ENCODER_STEM_TEST) \
		$(WHISPER_ENCODER_BLOCK_TEST) $(WHISPER_SMALL_ENCODER_CHECK) \
		$(WHISPER_SMALL_ENCODER_BENCH) $(WHISPER_SMALL_A113X_BENCH) \
		$(WHISPER_SMALL_A113X_CHECK) $(WHISPER_SMALL_DECODER_CHECK) \
		$(WHISPER_SMALL_A113X_DECODER_CHECK) $(WHISPER_SMALL_TRANSCRIBE) \
		$(WHISPER_SMALL_A113X_TRANSCRIBE)
