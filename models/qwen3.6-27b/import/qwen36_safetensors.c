#define _POSIX_C_SOURCE 200809L

#include "qwen36_safetensors.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { QWEN36_MAX_HEADER_BYTES = 256 * 1024 * 1024 };

static void set_error(char *output, size_t capacity, const char *format, ...) {
    if (output == NULL || capacity == 0) {
        return;
    }
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(output, capacity, format, arguments);
    va_end(arguments);
}

static int read_exact(int file, void *output, size_t length, off_t offset) {
    unsigned char *destination = output;
    size_t done = 0;
    while (done < length) {
        ssize_t amount = pread(file, destination + done, length - done,
                               offset + (off_t)done);
        if (amount < 0 && errno == EINTR) {
            continue;
        }
        if (amount <= 0) {
            return -1;
        }
        done += (size_t)amount;
    }
    return 0;
}

static uint64_t little_u64(const unsigned char input[8]) {
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index) {
        value |= (uint64_t)input[index] << (index * 8);
    }
    return value;
}

static const char *skip_space(const char *cursor, const char *end) {
    while (cursor < end && isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    return cursor;
}

static const char *find_field(const char *object_start, const char *object_end,
                              const char *field) {
    size_t field_length = strlen(field);
    for (const char *cursor = object_start; cursor + field_length + 2 < object_end;
         ++cursor) {
        if (*cursor == '"' &&
            (size_t)(object_end - cursor) > field_length + 2 &&
            memcmp(cursor + 1, field, field_length) == 0 &&
            cursor[field_length + 1] == '"') {
            const char *value = skip_space(cursor + field_length + 2, object_end);
            if (value < object_end && *value == ':') {
                return skip_space(value + 1, object_end);
            }
        }
    }
    return NULL;
}

static int parse_u64(const char **cursor_pointer, const char *end,
                     uint64_t *value) {
    const char *cursor = skip_space(*cursor_pointer, end);
    if (cursor >= end || !isdigit((unsigned char)*cursor)) {
        return -1;
    }
    uint64_t parsed = 0;
    while (cursor < end && isdigit((unsigned char)*cursor)) {
        unsigned digit = (unsigned)(*cursor - '0');
        if (parsed > (UINT64_MAX - digit) / 10) {
            return -1;
        }
        parsed = parsed * 10 + digit;
        ++cursor;
    }
    *cursor_pointer = cursor;
    *value = parsed;
    return 0;
}

static int parse_array(const char *cursor, const char *end,
                       uint64_t *values, size_t maximum, size_t *count) {
    cursor = skip_space(cursor, end);
    if (cursor >= end || *cursor++ != '[') {
        return -1;
    }
    size_t found = 0;
    for (;;) {
        cursor = skip_space(cursor, end);
        if (cursor >= end) {
            return -1;
        }
        if (*cursor == ']') {
            *count = found;
            return 0;
        }
        if (found >= maximum || parse_u64(&cursor, end, &values[found]) != 0) {
            return -1;
        }
        ++found;
        cursor = skip_space(cursor, end);
        if (cursor < end && *cursor == ',') {
            ++cursor;
            continue;
        }
        if (cursor < end && *cursor == ']') {
            *count = found;
            return 0;
        }
        return -1;
    }
}

static int locate_object(const char *header, size_t header_length,
                         const char *tensor_name,
                         const char **object_start, const char **object_end) {
    size_t name_length = strlen(tensor_name);
    const char *end = header + header_length;
    for (const char *cursor = header; cursor + name_length + 2 < end; ++cursor) {
        if (*cursor != '"' || memcmp(cursor + 1, tensor_name, name_length) != 0 ||
            cursor[name_length + 1] != '"') {
            continue;
        }
        const char *value = skip_space(cursor + name_length + 2, end);
        if (value >= end || *value++ != ':') {
            continue;
        }
        value = skip_space(value, end);
        if (value >= end || *value != '{') {
            continue;
        }
        const char *start = value;
        unsigned depth = 0;
        int in_string = 0;
        int escaped = 0;
        for (; value < end; ++value) {
            char character = *value;
            if (in_string) {
                if (escaped) {
                    escaped = 0;
                } else if (character == '\\') {
                    escaped = 1;
                } else if (character == '"') {
                    in_string = 0;
                }
                continue;
            }
            if (character == '"') {
                in_string = 1;
            } else if (character == '{') {
                ++depth;
            } else if (character == '}' && --depth == 0) {
                *object_start = start;
                *object_end = value + 1;
                return 0;
            }
        }
        return -1;
    }
    return 1;
}

int qwen36_safetensors_find(const char *path,
                            const char *tensor_name,
                            int require_payload,
                            qwen36_tensor_view *view,
                            char *error,
                            size_t error_capacity) {
    if (path == NULL || tensor_name == NULL || view == NULL) {
        set_error(error, error_capacity, "invalid safetensors arguments");
        return 1;
    }
    memset(view, 0, sizeof(*view));
    int file = open(path, O_RDONLY);
    if (file < 0) {
        set_error(error, error_capacity, "open %s: %s", path, strerror(errno));
        return 2;
    }
    struct stat status;
    if (fstat(file, &status) != 0 || status.st_size < 8) {
        set_error(error, error_capacity, "%s: missing safetensors header", path);
        close(file);
        return 3;
    }
    unsigned char length_bytes[8];
    if (read_exact(file, length_bytes, sizeof(length_bytes), 0) != 0) {
        set_error(error, error_capacity, "%s: cannot read header length", path);
        close(file);
        return 4;
    }
    uint64_t header_length = little_u64(length_bytes);
    if (header_length == 0 || header_length > QWEN36_MAX_HEADER_BYTES ||
        header_length > (uint64_t)status.st_size - 8) {
        set_error(error, error_capacity, "%s: invalid header length %" PRIu64,
                  path, header_length);
        close(file);
        return 5;
    }
    char *header = malloc((size_t)header_length + 1);
    if (header == NULL ||
        read_exact(file, header, (size_t)header_length, 8) != 0) {
        set_error(error, error_capacity, "%s: cannot allocate/read header", path);
        free(header);
        close(file);
        return 6;
    }
    header[header_length] = '\0';

    const char *object_start = NULL;
    const char *object_end = NULL;
    int locate_status = locate_object(header, (size_t)header_length, tensor_name,
                                      &object_start, &object_end);
    if (locate_status != 0) {
        set_error(error, error_capacity, "%s: tensor %s %s", path, tensor_name,
                  locate_status > 0 ? "not found" : "has malformed descriptor");
        free(header);
        close(file);
        return 7;
    }

    const char *dtype = find_field(object_start, object_end, "dtype");
    const char *shape = find_field(object_start, object_end, "shape");
    const char *offsets = find_field(object_start, object_end, "data_offsets");
    if (dtype == NULL || shape == NULL || offsets == NULL || *dtype++ != '"') {
        set_error(error, error_capacity, "%s: incomplete descriptor for %s",
                  path, tensor_name);
        free(header);
        close(file);
        return 8;
    }
    const char *dtype_end = memchr(dtype, '"', (size_t)(object_end - dtype));
    size_t dtype_length = dtype_end != NULL ? (size_t)(dtype_end - dtype) : 0;
    if (dtype_length == 0 || dtype_length >= sizeof(view->dtype)) {
        set_error(error, error_capacity, "%s: invalid dtype for %s", path, tensor_name);
        free(header);
        close(file);
        return 9;
    }
    memcpy(view->dtype, dtype, dtype_length);
    view->dtype[dtype_length] = '\0';
    if (parse_array(shape, object_end, view->shape,
                    sizeof(view->shape) / sizeof(view->shape[0]),
                    &view->rank) != 0) {
        set_error(error, error_capacity, "%s: invalid shape for %s", path, tensor_name);
        free(header);
        close(file);
        return 10;
    }
    uint64_t range[2];
    size_t range_count = 0;
    if (parse_array(offsets, object_end, range, 2, &range_count) != 0 ||
        range_count != 2 || range[1] < range[0]) {
        set_error(error, error_capacity, "%s: invalid data range for %s",
                  path, tensor_name);
        free(header);
        close(file);
        return 11;
    }
    uint64_t payload_start = 8 + header_length;
    if (range[0] > UINT64_MAX - payload_start ||
        range[1] > UINT64_MAX - payload_start) {
        set_error(error, error_capacity, "%s: data range overflow for %s",
                  path, tensor_name);
        free(header);
        close(file);
        return 12;
    }
    view->data_start = payload_start + range[0];
    view->data_length = range[1] - range[0];
    if (require_payload &&
        (view->data_start > (uint64_t)status.st_size ||
         view->data_length > (uint64_t)status.st_size - view->data_start)) {
        set_error(error, error_capacity,
                  "%s: payload for %s is truncated (need end %" PRIu64
                  ", file is %" PRIu64 ")",
                  path, tensor_name, view->data_start + view->data_length,
                  (uint64_t)status.st_size);
        free(header);
        close(file);
        return 13;
    }
    free(header);
    close(file);
    return 0;
}
