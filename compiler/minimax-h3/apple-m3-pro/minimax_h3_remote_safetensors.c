#include "minimax_h3_remote_safetensors.h"

#if !defined(MINIMAX_H3_LOCAL_ONLY)
#include <curl/curl.h>
#endif

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void remote_error(char *error, size_t capacity, const char *format, ...) {
    va_list arguments;
    if (error == NULL || capacity == 0u) return;
    va_start(arguments, format);
    vsnprintf(error, capacity, format, arguments);
    va_end(arguments);
}

#if !defined(MINIMAX_H3_LOCAL_ONLY)
typedef struct {
    unsigned char *bytes;
    size_t capacity;
    size_t length;
    int overflow;
} remote_sink;

static pthread_once_t remote_curl_once = PTHREAD_ONCE_INIT;

static void remote_curl_initialize(void) {
    (void)curl_global_init(CURL_GLOBAL_DEFAULT);
}

static size_t remote_write(void *data, size_t size, size_t count, void *opaque) {
    remote_sink *sink = opaque;
    size_t bytes;
    if (size != 0u && count > SIZE_MAX / size) {
        sink->overflow = 1;
        return 0u;
    }
    bytes = size * count;
    if (bytes > sink->capacity - sink->length) {
        sink->overflow = 1;
        return 0u;
    }
    memcpy(sink->bytes + sink->length, data, bytes);
    sink->length += bytes;
    return bytes;
}

static int remote_range_with_curl(CURL *curl,
                                  const char *url,
                                  uint64_t start,
                                  uint64_t length,
                                  void *output,
                                  size_t capacity,
                                  char *resolved_url,
                                  size_t resolved_capacity,
                                  char *error,
                                  size_t error_capacity) {
    CURLcode code;
    remote_sink sink;
    char range[96];
    long response = 0;

    if (url == NULL || output == NULL || length == 0u || length > capacity ||
        length > SIZE_MAX || UINT64_MAX - start < length - 1u) {
        remote_error(error, error_capacity, "invalid HTTP range");
        return 2;
    }
    if (snprintf(range, sizeof(range), "%" PRIu64 "-%" PRIu64, start,
                 start + length - 1u) >= (int)sizeof(range)) {
        remote_error(error, error_capacity, "HTTP range is too long");
        return 2;
    }
    memset(&sink, 0, sizeof(sink));
    sink.bytes = output;
    sink.capacity = capacity;
    if (curl == NULL) return 1;
    curl_easy_reset(curl);
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_RANGE, range);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 8L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "cpu-llms-in-c/minimax-h3");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, remote_write);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &sink);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    code = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response);
    if (code != CURLE_OK) {
        remote_error(error, error_capacity, "HTTP range failed: %s%s",
                     curl_easy_strerror(code),
                     sink.overflow ? " (server ignored bounded range)" : "");
        return 1;
    }
    if (response != 206L) {
        remote_error(error, error_capacity,
                     "HTTP server returned %ld, expected 206", response);
        return 1;
    }
    if (sink.length != (size_t)length) {
        remote_error(error, error_capacity,
                     "short HTTP range: got %zu, expected %" PRIu64,
                     sink.length, length);
        return 1;
    }
    if (resolved_url != NULL && resolved_capacity != 0u) {
        char *effective = NULL;
        curl_easy_getinfo(curl, CURLINFO_EFFECTIVE_URL, &effective);
        if (effective == NULL || strlen(effective) >= resolved_capacity) {
            remote_error(error, error_capacity,
                         "resolved HTTP URL exceeds internal capacity");
            return 1;
        }
        snprintf(resolved_url, resolved_capacity, "%s", effective);
    }
    return 0;
}
#endif

static int remote_range(const char *url,
                        uint64_t start,
                        uint64_t length,
                        void *output,
                        size_t capacity,
                        char *resolved_url,
                        size_t resolved_capacity,
                        char *error,
                        size_t error_capacity) {
    if (url != NULL && strncmp(url, "file://", 7u) == 0) {
        const char *path = url + 7u;
        if (path[0] != '/' || start > INT64_MAX ||
            length > (uint64_t)INT64_MAX - start || length > capacity) {
            remote_error(error, error_capacity, "invalid local file range");
            return 2;
        }
        int descriptor = open(path, O_RDONLY);
        if (descriptor < 0) {
            remote_error(error, error_capacity, "cannot open local weight: %s",
                         strerror(errno));
            return 1;
        }
        size_t completed = 0u;
        while (completed < (size_t)length) {
            ssize_t count = pread(descriptor,
                                  (unsigned char *)output + completed,
                                  (size_t)length - completed,
                                  (off_t)(start + completed));
            if (count <= 0) {
                int saved_errno = errno;
                close(descriptor);
                remote_error(error, error_capacity,
                             count == 0 ? "short local file range"
                                        : "local file range failed: %s",
                             count == 0 ? "" : strerror(saved_errno));
                return 1;
            }
            completed += (size_t)count;
        }
        close(descriptor);
        if (resolved_url != NULL && resolved_capacity != 0u) {
            if (strlen(url) >= resolved_capacity) {
                remote_error(error, error_capacity,
                             "local weight URL exceeds internal capacity");
                return 1;
            }
            snprintf(resolved_url, resolved_capacity, "%s", url);
        }
        return 0;
    }
#if defined(MINIMAX_H3_LOCAL_ONLY)
    remote_error(error, error_capacity,
                 "network safetensors reads are disabled in this runtime");
    return 1;
#else
    pthread_once(&remote_curl_once, remote_curl_initialize);
    CURL *curl = curl_easy_init();
    if (curl == NULL) {
        remote_error(error, error_capacity, "curl_easy_init failed");
        return 1;
    }
    int status = remote_range_with_curl(curl, url, start, length, output,
                                        capacity, resolved_url,
                                        resolved_capacity, error,
                                        error_capacity);
    curl_easy_cleanup(curl);
    return status;
#endif
}

#if !defined(MINIMAX_H3_LOCAL_ONLY)
typedef struct {
    const minimax_h3_remote_safetensors *file;
    unsigned char *bytes;
    size_t chunk_bytes;
    uint64_t range_start;
    uint64_t range_length;
    uint64_t next;
    uint64_t completed;
    int failed;
    pthread_mutex_t mutex;
    minimax_h3_remote_progress_fn progress;
    void *progress_context;
    char message[512];
} remote_materialize_state;

static void *remote_materialize_worker(void *opaque) {
    remote_materialize_state *state = opaque;
    pthread_once(&remote_curl_once, remote_curl_initialize);
    CURL *curl = curl_easy_init();
    if (curl == NULL) {
        pthread_mutex_lock(&state->mutex);
        state->failed = 1;
        snprintf(state->message, sizeof(state->message),
                 "curl_easy_init failed in range worker");
        pthread_mutex_unlock(&state->mutex);
        return NULL;
    }
    for (;;) {
        uint64_t relative_start;
        uint64_t absolute_start;
        size_t current;
        pthread_mutex_lock(&state->mutex);
        if (state->failed || state->next >= state->range_length) {
            pthread_mutex_unlock(&state->mutex);
            break;
        }
        relative_start = state->next;
        uint64_t remaining = state->range_length - relative_start;
        current = remaining < state->chunk_bytes ? (size_t)remaining
                                                 : state->chunk_bytes;
        state->next += current;
        pthread_mutex_unlock(&state->mutex);

        absolute_start = state->range_start + relative_start;

        char local_error[512];
        int range_status = remote_range_with_curl(
            curl, state->file->url, absolute_start, current,
            state->bytes + (size_t)relative_start, current, NULL, 0u,
            local_error, sizeof(local_error));
        if (range_status != 0 && state->file->source_url[0] != '\0') {
            range_status = remote_range(
                state->file->source_url, absolute_start, current,
                state->bytes + (size_t)relative_start, current, NULL, 0u,
                local_error, sizeof(local_error));
        }
        if (range_status != 0) {
            pthread_mutex_lock(&state->mutex);
            if (!state->failed) {
                state->failed = 1;
                snprintf(state->message, sizeof(state->message), "%s",
                         local_error);
            }
            pthread_mutex_unlock(&state->mutex);
            break;
        }
        pthread_mutex_lock(&state->mutex);
        state->completed += current;
        if (state->progress != NULL)
            state->progress(state->completed, state->range_length,
                            state->progress_context);
        pthread_mutex_unlock(&state->mutex);
    }
    curl_easy_cleanup(curl);
    return NULL;
}

static int remote_parallel_ranges(remote_materialize_state *state,
                                  char *error,
                                  size_t error_capacity) {
    enum { worker_count = 12 };
    pthread_t threads[worker_count];
    size_t started = 0u;
    for (; started < worker_count; ++started) {
        if (pthread_create(&threads[started], NULL,
                           remote_materialize_worker, state) != 0) {
            pthread_mutex_lock(&state->mutex);
            state->failed = 1;
            snprintf(state->message, sizeof(state->message),
                     "cannot start range worker %zu", started);
            pthread_mutex_unlock(&state->mutex);
            break;
        }
    }
    for (size_t index = 0u; index < started; ++index)
        pthread_join(threads[index], NULL);
    pthread_mutex_destroy(&state->mutex);
    if (state->failed || state->completed != state->range_length) {
        remote_error(error, error_capacity, "%s",
                     state->message[0] != '\0' ? state->message
                                                : "incomplete parallel range read");
        return 1;
    }
    return 0;
}
#endif

static uint64_t little_u64(const unsigned char bytes[8]) {
    uint64_t value = 0u;
    for (unsigned index = 0u; index < 8u; ++index)
        value |= (uint64_t)bytes[index] << (index * 8u);
    return value;
}

static int remote_payload_length(const char *header, uint64_t *length) {
    const char *cursor = header;
    uint64_t maximum = 0u;
    size_t count = 0u;
    while ((cursor = strstr(cursor, "\"data_offsets\"")) != NULL) {
        uint64_t start;
        uint64_t end;
        cursor = strchr(cursor, '[');
        if (cursor == NULL) return 0;
        ++cursor;
        while (isspace((unsigned char)*cursor)) ++cursor;
        char *parsed_end;
        errno = 0;
        start = strtoull(cursor, &parsed_end, 10);
        if (errno != 0 || parsed_end == cursor) return 0;
        cursor = strchr(parsed_end, ',');
        if (cursor == NULL) return 0;
        ++cursor;
        errno = 0;
        end = strtoull(cursor, &parsed_end, 10);
        if (errno != 0 || parsed_end == cursor || end < start) return 0;
        if (end > maximum) maximum = end;
        cursor = parsed_end;
        ++count;
    }
    if (count == 0u || maximum == 0u) return 0;
    *length = maximum;
    return 1;
}

int minimax_h3_remote_safetensors_open(
    const char *url,
    minimax_h3_remote_safetensors *file,
    char *error,
    size_t error_capacity) {
    minimax_h3_remote_safetensors result;
    unsigned char prefix[12];
    uint64_t header_length;
    uint64_t leading_padding = 0u;
    int status;

    if (url == NULL || file == NULL || strlen(url) >= sizeof(result.url)) {
        remote_error(error, error_capacity, "invalid remote safetensors URL");
        return 2;
    }
    memset(&result, 0, sizeof(result));
    snprintf(result.source_url, sizeof(result.source_url), "%s", url);
    snprintf(result.url, sizeof(result.url), "%s", url);
    status = remote_range(result.url, 0u, sizeof(prefix), prefix,
                          sizeof(prefix), result.url, sizeof(result.url), error,
                          error_capacity);
    if (status != 0) return status;
    header_length = 0u;
    for (size_t padding = 0u; padding <= 3u; ++padding) {
        uint64_t candidate = little_u64(prefix + padding);
        if (candidate != 0u &&
            candidate <= UINT64_C(64) * 1024u * 1024u &&
            prefix[padding + 8u] == '{') {
            leading_padding = padding;
            header_length = candidate;
            break;
        }
    }
    if (header_length == 0u || header_length > SIZE_MAX - 1u ||
        header_length > UINT64_C(64) * 1024u * 1024u) {
        remote_error(error, error_capacity,
                     "invalid safetensors header length: %" PRIu64,
                     header_length);
        return 1;
    }
    result.header = malloc((size_t)header_length + 1u);
    if (result.header == NULL) {
        remote_error(error, error_capacity, "header allocation failed");
        return 1;
    }
    status = remote_range(result.url, leading_padding + 8u, header_length,
                          result.header,
                          (size_t)header_length, NULL, 0u, error,
                          error_capacity);
    if (status != 0) {
        free(result.header);
        return status;
    }
    result.header[header_length] = '\0';
    result.header_length = (size_t)header_length;
    result.payload_start = leading_padding + 8u + header_length;
    {
        uint64_t payload_length = 0u;
        if (!remote_payload_length(result.header, &payload_length) ||
            UINT64_MAX - result.payload_start < payload_length) {
            free(result.header);
            remote_error(error, error_capacity,
                         "cannot determine safetensors payload length");
            return 1;
        }
        result.file_length = result.payload_start + payload_length;
    }
    *file = result;
    return 0;
}

void minimax_h3_remote_safetensors_close(
    minimax_h3_remote_safetensors *file) {
    if (file == NULL) return;
    free(file->header);
    memset(file, 0, sizeof(*file));
}

static const char *skip_space(const char *cursor) {
    while (*cursor != '\0' && isspace((unsigned char)*cursor)) ++cursor;
    return cursor;
}

static int parse_u64(const char **cursor, uint64_t *value) {
    char *end;
    unsigned long long parsed;
    const char *start = skip_space(*cursor);
    errno = 0;
    parsed = strtoull(start, &end, 10);
    if (errno != 0 || end == start) return 0;
    *cursor = end;
    *value = (uint64_t)parsed;
    return 1;
}

static const char *find_field(const char *object, const char *field) {
    char pattern[64];
    if (snprintf(pattern, sizeof(pattern), "\"%s\"", field) >=
        (int)sizeof(pattern)) return NULL;
    return strstr(object, pattern);
}

int minimax_h3_remote_safetensors_find(
    const minimax_h3_remote_safetensors *file,
    const char *tensor_name,
    minimax_h3_remote_tensor *tensor,
    char *error,
    size_t error_capacity) {
    minimax_h3_remote_tensor result;
    char *pattern;
    size_t pattern_length;
    const char *object;
    const char *cursor;
    const char *end;
    uint64_t offsets[2];

    if (file == NULL || file->header == NULL || tensor_name == NULL ||
        tensor == NULL || strchr(tensor_name, '"') != NULL) {
        remote_error(error, error_capacity, "invalid tensor lookup");
        return 2;
    }
    pattern_length = strlen(tensor_name) + 3u;
    pattern = malloc(pattern_length);
    if (pattern == NULL) {
        remote_error(error, error_capacity, "tensor pattern allocation failed");
        return 1;
    }
    snprintf(pattern, pattern_length, "\"%s\"", tensor_name);
    object = strstr(file->header, pattern);
    free(pattern);
    if (object == NULL) {
        remote_error(error, error_capacity, "tensor not found: %s", tensor_name);
        return 1;
    }
    memset(&result, 0, sizeof(result));
    cursor = find_field(object, "dtype");
    if (cursor == NULL || (cursor = strchr(cursor, ':')) == NULL ||
        (cursor = strchr(cursor, '"')) == NULL) {
        remote_error(error, error_capacity, "missing dtype: %s", tensor_name);
        return 1;
    }
    ++cursor;
    end = strchr(cursor, '"');
    if (end == NULL || (size_t)(end - cursor) >= sizeof(result.dtype)) {
        remote_error(error, error_capacity, "invalid dtype: %s", tensor_name);
        return 1;
    }
    memcpy(result.dtype, cursor, (size_t)(end - cursor));
    result.dtype[end - cursor] = '\0';

    cursor = find_field(object, "shape");
    if (cursor == NULL || (cursor = strchr(cursor, '[')) == NULL) {
        remote_error(error, error_capacity, "missing shape: %s", tensor_name);
        return 1;
    }
    ++cursor;
    for (;;) {
        cursor = skip_space(cursor);
        if (*cursor == ']') break;
        if (result.rank >= MINIMAX_H3_REMOTE_MAX_RANK ||
            !parse_u64(&cursor, &result.shape[result.rank++])) {
            remote_error(error, error_capacity, "invalid shape: %s", tensor_name);
            return 1;
        }
        cursor = skip_space(cursor);
        if (*cursor == ',') ++cursor;
        else if (*cursor != ']') {
            remote_error(error, error_capacity, "invalid shape list: %s",
                         tensor_name);
            return 1;
        }
    }

    cursor = find_field(object, "data_offsets");
    if (cursor == NULL || (cursor = strchr(cursor, '[')) == NULL) {
        remote_error(error, error_capacity, "missing data offsets: %s",
                     tensor_name);
        return 1;
    }
    ++cursor;
    if (!parse_u64(&cursor, &offsets[0]) ||
        (cursor = strchr(cursor, ',')) == NULL) {
        remote_error(error, error_capacity, "invalid data offsets: %s",
                     tensor_name);
        return 1;
    }
    ++cursor;
    if (!parse_u64(&cursor, &offsets[1]) || offsets[1] < offsets[0] ||
        UINT64_MAX - file->payload_start < offsets[0]) {
        remote_error(error, error_capacity, "invalid data range: %s",
                     tensor_name);
        return 1;
    }
    result.data_start = file->payload_start + offsets[0];
    result.data_length = offsets[1] - offsets[0];
    *tensor = result;
    return 0;
}

int minimax_h3_remote_safetensors_fetch(
    const minimax_h3_remote_safetensors *file,
    const minimax_h3_remote_tensor *tensor,
    void *output,
    size_t output_capacity,
    char *error,
    size_t error_capacity) {
    if (file == NULL || tensor == NULL || output == NULL ||
        tensor->data_length != output_capacity) {
        remote_error(error, error_capacity,
                     "tensor output capacity must equal data length");
        return 2;
    }
    return remote_range(file->url, tensor->data_start, tensor->data_length,
                        output, output_capacity, NULL, 0u, error,
                        error_capacity);
}

int minimax_h3_remote_safetensors_fetch_parallel(
    const minimax_h3_remote_safetensors *file,
    const minimax_h3_remote_tensor *tensor,
    void *output,
    size_t output_capacity,
    size_t chunk_bytes,
    char *error,
    size_t error_capacity) {
    if (file == NULL || file->header == NULL || tensor == NULL ||
        output == NULL || tensor->data_length == 0u ||
        tensor->data_length != output_capacity ||
        tensor->data_length > SIZE_MAX ||
        UINT64_MAX - tensor->data_start < tensor->data_length) {
        remote_error(error, error_capacity,
                     "parallel tensor output capacity must equal data length");
        return 2;
    }
    if (chunk_bytes == 0u) chunk_bytes = 16u * 1024u * 1024u;
#if defined(MINIMAX_H3_LOCAL_ONLY)
    (void)chunk_bytes;
    return remote_range(file->url, tensor->data_start, tensor->data_length,
                        output, output_capacity, NULL, 0u, error,
                        error_capacity);
#else
    remote_materialize_state state = {
        .file = file,
        .bytes = output,
        .chunk_bytes = chunk_bytes,
        .range_start = tensor->data_start,
        .range_length = tensor->data_length,
        .mutex = PTHREAD_MUTEX_INITIALIZER,
    };
    return remote_parallel_ranges(&state, error, error_capacity);
#endif
}

int minimax_h3_remote_safetensors_read(
    const minimax_h3_remote_safetensors *file,
    uint64_t absolute_start,
    void *output,
    size_t length,
    char *error,
    size_t error_capacity) {
    if (file == NULL || file->header == NULL || output == NULL || length == 0u) {
        remote_error(error, error_capacity, "invalid absolute remote read");
        return 2;
    }
    return remote_range(file->url, absolute_start, length, output, length,
                        NULL, 0u, error, error_capacity);
}

int minimax_h3_remote_safetensors_materialize(
    const minimax_h3_remote_safetensors *file,
    void *output,
    size_t output_capacity,
    size_t chunk_bytes,
    minimax_h3_remote_progress_fn progress,
    void *progress_context,
    char *error,
    size_t error_capacity) {
    if (file == NULL || file->header == NULL || output == NULL ||
        file->file_length == 0u || file->file_length > output_capacity ||
        file->file_length > SIZE_MAX) {
        remote_error(error, error_capacity,
                     "materialization buffer does not match remote file");
        return 2;
    }
    if (chunk_bytes == 0u) chunk_bytes = 32u * 1024u * 1024u;
#if defined(MINIMAX_H3_LOCAL_ONLY)
    (void)chunk_bytes;
    int status = remote_range(file->url, 0u, file->file_length, output,
                              output_capacity, NULL, 0u, error,
                              error_capacity);
    if (status == 0 && progress != NULL)
        progress(file->file_length, file->file_length, progress_context);
    return status;
#else
    remote_materialize_state state = {
        .file = file,
        .bytes = output,
        .chunk_bytes = chunk_bytes,
        .range_start = 0u,
        .range_length = file->file_length,
        .progress = progress,
        .progress_context = progress_context,
        .mutex = PTHREAD_MUTEX_INITIALIZER,
    };
    return remote_parallel_ranges(&state, error, error_capacity);
#endif
}
