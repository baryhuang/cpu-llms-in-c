CC ?= cc

CPPFLAGS ?=
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS ?= -pthread

BUILD_DIR := build
TARGET_PROBE := $(BUILD_DIR)/target-probe

.PHONY: all clean

all: $(TARGET_PROBE)

$(TARGET_PROBE): bench/target_probe.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET_PROBE)
