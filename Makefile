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

.PHONY: all clean fixture linux-tools test

all: $(GEMMA4_LAYER_TEST) $(GEMMA4_TASK)

linux-tools: $(TARGET_PROBE)

fixture: $(GEMMA4_LAYER_FIXTURE)

test: $(GEMMA4_LAYER_TEST) $(GEMMA4_LAYER_FIXTURE)
	python3 -m unittest discover -s tests -p 'test_*.py'
	$(GEMMA4_LAYER_TEST) $(GEMMA4_LAYER_FIXTURE)

$(TARGET_PROBE): runtime/target_probe.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

$(GEMMA4_LAYER_TEST): tests/gemma4_layer_test.c runtime/gemma4_layer.c runtime/gemma4_layer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Iruntime tests/gemma4_layer_test.c runtime/gemma4_layer.c \
		-o $@ $(LDFLAGS) -lm

$(GEMMA4_TASK): runtime/gemma4_task.c runtime/gemma4_task.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -Iruntime runtime/gemma4_task.c \
		-o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(GEMMA4_LAYER_FIXTURE): compiler/generate_gemma4_layer_fixture.py
	python3 $< --output $@

clean:
	rm -f $(TARGET_PROBE) $(GEMMA4_LAYER_TEST) $(GEMMA4_TASK)
