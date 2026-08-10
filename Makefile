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

clean:
	rm -f $(TARGET_PROBE) $(GEMMA4_LAYER_TEST) $(GEMMA4_TASK)
