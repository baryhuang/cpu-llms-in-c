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
WHISPER_GENERIC := models/whisper-large-v3/targets/generic
WHISPER_LOG_MEL_TEST := $(BUILD_DIR)/whisper-log-mel-test
WHISPER_LOG_MEL_FIXTURE := tests/fixtures/whisper_log_mel_128_v1.bin

.PHONY: all a113x clean fixture linux-tools test

all: $(GEMMA4_LAYER_TEST) $(GEMMA4_TASK)

linux-tools: $(TARGET_PROBE)

a113x: $(QWEN35_A113X_TASK)

fixture: $(GEMMA4_LAYER_FIXTURE)

test: $(GEMMA4_LAYER_TEST) $(GEMMA4_LAYER_FIXTURE) $(QWEN35_LAYER_TEST) $(QWEN35_LAYER_FIXTURE) \
	$(WHISPER_LOG_MEL_TEST) $(WHISPER_LOG_MEL_FIXTURE)
	python3 -m unittest discover -s tests -p 'test_*.py'
	$(GEMMA4_LAYER_TEST) $(GEMMA4_LAYER_FIXTURE)
	$(QWEN35_LAYER_TEST) $(QWEN35_LAYER_FIXTURE)
	$(WHISPER_LOG_MEL_TEST) $(WHISPER_LOG_MEL_FIXTURE)

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

$(WHISPER_LOG_MEL_TEST): tests/whisper_log_mel_test.c $(WHISPER_GENERIC)/whisper_frontend.c $(WHISPER_GENERIC)/whisper_frontend.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_GENERIC) tests/whisper_log_mel_test.c \
		$(WHISPER_GENERIC)/whisper_frontend.c -o $@ $(LDFLAGS) -lm

$(WHISPER_LOG_MEL_FIXTURE): compiler/generate_whisper_log_mel_fixture.py
	python3 $< --output $@

clean:
	rm -f $(TARGET_PROBE) $(GEMMA4_LAYER_TEST) $(GEMMA4_TASK) $(QWEN35_LAYER_TEST) \
		$(QWEN35_TASK) $(QWEN35_A113X_TASK) $(WHISPER_LOG_MEL_TEST)
