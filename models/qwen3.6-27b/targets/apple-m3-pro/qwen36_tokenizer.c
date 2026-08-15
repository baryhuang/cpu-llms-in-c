#define _POSIX_C_SOURCE 200809L

#include "qwen36_tokenizer.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <CoreFoundation/CoreFoundation.h>
#include <unicode/uchar.h>
#include <unicode/utf8.h>

struct qwen36_tokenizer {
    int file;
    size_t mapped_bytes;
    const unsigned char *mapping;
    const qwen36_tokenizer_image_header *header;
    const qwen36_token_directory_entry *directory;
    const unsigned char *token_blob;
    const qwen36_token_merge_entry *merges;
};

typedef struct {
    uint32_t *ids;
    size_t count;
    size_t capacity;
} token_vector;

static void tokenizer_error(char *message, size_t capacity,
                            const char *format, ...) {
    if (message == NULL || capacity == 0) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(message, capacity, format, arguments);
    va_end(arguments);
}

static int checked_range(uint64_t offset, uint64_t length,
                         uint64_t file_bytes) {
    return offset <= file_bytes && length <= file_bytes - offset;
}

qwen36_tokenizer *qwen36_tokenizer_open(
    const char *image_path, char *error_message, size_t error_capacity) {
    if (image_path == NULL) {
        tokenizer_error(error_message, error_capacity,
                        "tokenizer image path is null");
        return NULL;
    }
    int file = open(image_path, O_RDONLY);
    struct stat status;
    if (file < 0 || fstat(file, &status) != 0 ||
        status.st_size < QWEN36_TOKENIZER_HEADER_BYTES) {
        if (file >= 0) close(file);
        tokenizer_error(error_message, error_capacity,
                        "cannot open tokenizer image: %s", strerror(errno));
        return NULL;
    }
    size_t bytes = (size_t)status.st_size;
    const unsigned char *mapping =
        mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, file, 0);
    if (mapping == MAP_FAILED) {
        close(file);
        tokenizer_error(error_message, error_capacity,
                        "cannot map tokenizer image: %s", strerror(errno));
        return NULL;
    }
    const qwen36_tokenizer_image_header *header =
        (const qwen36_tokenizer_image_header *)mapping;
    int valid =
        memcmp(header->magic, QWEN36_TOKENIZER_MAGIC, 8) == 0 &&
        header->version == QWEN36_TOKENIZER_VERSION &&
        header->header_bytes == QWEN36_TOKENIZER_HEADER_BYTES &&
        header->vocab_size == QWEN36_TOKENIZER_VOCAB &&
        header->base_vocab_size == QWEN36_TOKENIZER_BASE_VOCAB &&
        header->added_first == QWEN36_TOKENIZER_ADDED_FIRST &&
        header->added_count == QWEN36_TOKENIZER_ADDED_COUNT &&
        header->merge_count == QWEN36_TOKENIZER_MERGES &&
        header->token_entry_bytes ==
            sizeof(qwen36_token_directory_entry) &&
        (memcmp(header->source_sha256, QWEN36_TOKENIZER_SOURCE_SHA256,
                64) == 0 ||
         memcmp(header->source_sha256, QWEN38_TOKENIZER_SOURCE_SHA256,
                64) == 0) &&
        header->token_directory_bytes ==
            (uint64_t)QWEN36_TOKENIZER_VOCAB *
                sizeof(qwen36_token_directory_entry) &&
        header->merges_bytes ==
            (uint64_t)QWEN36_TOKENIZER_MERGES *
                sizeof(qwen36_token_merge_entry) &&
        checked_range(header->token_directory_offset,
                      header->token_directory_bytes, bytes) &&
        checked_range(header->token_blob_offset,
                      header->token_blob_bytes, bytes) &&
        checked_range(header->merges_offset, header->merges_bytes, bytes);
    if (!valid) {
        munmap((void *)mapping, bytes);
        close(file);
        tokenizer_error(error_message, error_capacity,
                        "invalid or incompatible tokenizer image");
        return NULL;
    }
    const qwen36_token_directory_entry *directory =
        (const qwen36_token_directory_entry *)(mapping +
            header->token_directory_offset);
    for (uint32_t id = 0; id < header->vocab_size; ++id) {
        if (!checked_range(directory[id].offset, directory[id].length,
                           header->token_blob_bytes)) {
            munmap((void *)mapping, bytes);
            close(file);
            tokenizer_error(error_message, error_capacity,
                            "token directory entry %u is out of range", id);
            return NULL;
        }
    }
    qwen36_tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (tokenizer == NULL) {
        munmap((void *)mapping, bytes);
        close(file);
        tokenizer_error(error_message, error_capacity,
                        "cannot allocate tokenizer runtime");
        return NULL;
    }
    tokenizer->file = file;
    tokenizer->mapped_bytes = bytes;
    tokenizer->mapping = mapping;
    tokenizer->header = header;
    tokenizer->directory = directory;
    tokenizer->token_blob = mapping + header->token_blob_offset;
    tokenizer->merges = (const qwen36_token_merge_entry *)(mapping +
                            header->merges_offset);
    return tokenizer;
}

void qwen36_tokenizer_close(qwen36_tokenizer *tokenizer) {
    if (tokenizer == NULL) return;
    if (tokenizer->mapping != NULL)
        munmap((void *)tokenizer->mapping, tokenizer->mapped_bytes);
    if (tokenizer->file >= 0) close(tokenizer->file);
    free(tokenizer);
}

static int vector_reserve(token_vector *vector, size_t requested) {
    if (requested <= vector->capacity) return 0;
    size_t capacity = vector->capacity == 0 ? 64 : vector->capacity;
    while (capacity < requested) {
        if (capacity > SIZE_MAX / 2) return -1;
        capacity *= 2;
    }
    uint32_t *ids = realloc(vector->ids, capacity * sizeof(*ids));
    if (ids == NULL) return -1;
    vector->ids = ids;
    vector->capacity = capacity;
    return 0;
}

static int vector_append(token_vector *vector, uint32_t id) {
    if (vector_reserve(vector, vector->count + 1) != 0) return -1;
    vector->ids[vector->count++] = id;
    return 0;
}

static const qwen36_token_merge_entry *find_merge(
    const qwen36_tokenizer *tokenizer, uint32_t left, uint32_t right) {
    uint64_t key = ((uint64_t)left << 32) | right;
    size_t low = 0;
    size_t high = tokenizer->header->merge_count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        const qwen36_token_merge_entry *entry = &tokenizer->merges[middle];
        uint64_t candidate = ((uint64_t)entry->left << 32) | entry->right;
        if (candidate < key) low = middle + 1;
        else high = middle;
    }
    if (low == tokenizer->header->merge_count) return NULL;
    const qwen36_token_merge_entry *entry = &tokenizer->merges[low];
    return entry->left == left && entry->right == right ? entry : NULL;
}

static int append_bpe_piece(const qwen36_tokenizer *tokenizer,
                            const unsigned char *bytes, size_t length,
                            token_vector *output) {
    if (length == 0) return 0;
    uint32_t *piece = malloc(length * sizeof(*piece));
    if (piece == NULL) return -1;
    for (size_t index = 0; index < length; ++index)
        piece[index] = tokenizer->header->byte_token_ids[bytes[index]];
    size_t count = length;
    for (;;) {
        uint32_t best_rank = UINT32_MAX;
        uint32_t best_left = 0;
        uint32_t best_right = 0;
        uint32_t best_result = 0;
        for (size_t index = 0; index + 1 < count; ++index) {
            const qwen36_token_merge_entry *merge =
                find_merge(tokenizer, piece[index], piece[index + 1]);
            if (merge != NULL && merge->rank < best_rank) {
                best_rank = merge->rank;
                best_left = merge->left;
                best_right = merge->right;
                best_result = merge->result;
            }
        }
        if (best_rank == UINT32_MAX) break;
        size_t read_index = 0;
        size_t write_index = 0;
        while (read_index < count) {
            if (read_index + 1 < count &&
                piece[read_index] == best_left &&
                piece[read_index + 1] == best_right) {
                piece[write_index++] = best_result;
                read_index += 2;
            } else {
                piece[write_index++] = piece[read_index++];
            }
        }
        count = write_index;
    }
    if (vector_reserve(output, output->count + count) != 0) {
        free(piece);
        return -1;
    }
    memcpy(output->ids + output->count, piece, count * sizeof(*piece));
    output->count += count;
    free(piece);
    return 0;
}

static int normalize_nfc(const unsigned char *input, size_t input_length,
                         unsigned char **output, size_t *output_length) {
    CFStringRef source = CFStringCreateWithBytes(
        kCFAllocatorDefault, input, (CFIndex)input_length,
        kCFStringEncodingUTF8, false);
    if (source == NULL) return -1;
    CFMutableStringRef normalized = CFStringCreateMutableCopy(
        kCFAllocatorDefault, 0, source);
    CFRelease(source);
    if (normalized == NULL) return -1;
    CFStringNormalize(normalized, kCFStringNormalizationFormC);
    CFIndex characters = CFStringGetLength(normalized);
    CFIndex maximum = CFStringGetMaximumSizeForEncoding(
        characters, kCFStringEncodingUTF8);
    if (maximum < 0) {
        CFRelease(normalized);
        return -1;
    }
    unsigned char *utf8 = malloc((size_t)maximum + 1);
    if (utf8 == NULL || !CFStringGetCString(normalized, (char *)utf8,
                                             maximum + 1,
                                             kCFStringEncodingUTF8)) {
        free(utf8);
        CFRelease(normalized);
        return -1;
    }
    CFRelease(normalized);
    *output_length = strlen((const char *)utf8);
    *output = utf8;
    return 0;
}

typedef struct {
    UChar32 codepoint;
    size_t end;
    int letter;
    int mark;
    int number;
    int whitespace;
} unicode_unit;

static int read_unit(const unsigned char *text, size_t length,
                     size_t position, unicode_unit *unit) {
    if (position >= length) return -1;
    int32_t index = (int32_t)position;
    UChar32 codepoint;
    U8_NEXT(text, index, (int32_t)length, codepoint);
    if (codepoint < 0) return -1;
    int8_t category = u_charType(codepoint);
    unit->codepoint = codepoint;
    unit->end = (size_t)index;
    unit->letter = category == U_UPPERCASE_LETTER ||
                   category == U_LOWERCASE_LETTER ||
                   category == U_TITLECASE_LETTER ||
                   category == U_MODIFIER_LETTER ||
                   category == U_OTHER_LETTER;
    unit->mark = category == U_NON_SPACING_MARK ||
                 category == U_ENCLOSING_MARK ||
                 category == U_COMBINING_SPACING_MARK;
    unit->number = category == U_DECIMAL_DIGIT_NUMBER ||
                   category == U_LETTER_NUMBER ||
                   category == U_OTHER_NUMBER;
    unit->whitespace = u_isUWhiteSpace(codepoint);
    return 0;
}

static int ascii_equal_fold(const unsigned char *text, size_t available,
                            const char *pattern) {
    size_t length = strlen(pattern);
    if (length > available) return 0;
    for (size_t index = 0; index < length; ++index) {
        unsigned char value = text[index];
        if (value >= 'A' && value <= 'Z') value += 'a' - 'A';
        if (value != (unsigned char)pattern[index]) return 0;
    }
    return (int)length;
}

static int is_punctuation_unit(const unicode_unit *unit) {
    return !unit->whitespace && !unit->letter &&
           !unit->mark && !unit->number;
}

static int next_piece(const unsigned char *text, size_t length,
                      size_t start, size_t *end) {
    static const char *contractions[] = {
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"
    };
    if (text[start] == '\'') {
        for (size_t index = 0;
             index < sizeof(contractions) / sizeof(contractions[0]);
             ++index) {
            int matched = ascii_equal_fold(text + start, length - start,
                                           contractions[index]);
            if (matched != 0) {
                *end = start + (size_t)matched;
                return 0;
            }
        }
    }
    unicode_unit first;
    if (read_unit(text, length, start, &first) != 0) return -1;
    size_t cursor = start;
    unicode_unit current = first;
    int has_prefix = !first.letter && !first.number &&
                     first.codepoint != '\r' && first.codepoint != '\n';
    if (first.letter || first.mark) {
        cursor = first.end;
    } else if (has_prefix && first.end < length) {
        unicode_unit next;
        if (read_unit(text, length, first.end, &next) == 0 &&
            (next.letter || next.mark)) {
            cursor = next.end;
        }
    }
    if (cursor != start) {
        while (cursor < length &&
               read_unit(text, length, cursor, &current) == 0 &&
               (current.letter || current.mark)) cursor = current.end;
        *end = cursor;
        return 0;
    }
    if (first.number) {
        *end = first.end;
        return 0;
    }
    cursor = start;
    if (first.codepoint == ' ' && first.end < length) {
        unicode_unit next;
        if (read_unit(text, length, first.end, &next) == 0 &&
            is_punctuation_unit(&next)) cursor = first.end;
    }
    if (cursor != start || is_punctuation_unit(&first)) {
        if (cursor == start) cursor = first.end;
        while (cursor < length &&
               read_unit(text, length, cursor, &current) == 0 &&
               is_punctuation_unit(&current)) cursor = current.end;
        while (cursor < length &&
               read_unit(text, length, cursor, &current) == 0 &&
               (current.codepoint == '\r' || current.codepoint == '\n'))
            cursor = current.end;
        *end = cursor;
        return 0;
    }
    if (first.whitespace) {
        cursor = start;
        size_t last_newline = 0;
        while (cursor < length &&
               read_unit(text, length, cursor, &current) == 0 &&
               current.whitespace) {
            cursor = current.end;
            if (current.codepoint == '\r' || current.codepoint == '\n')
                last_newline = cursor;
        }
        *end = last_newline != 0 ? last_newline : cursor;
        return 0;
    }
    *end = first.end;
    return 0;
}

static int encode_regular(const qwen36_tokenizer *tokenizer,
                          const unsigned char *input, size_t input_length,
                          token_vector *output) {
    if (input_length == 0) return 0;
    unsigned char *normalized = NULL;
    size_t normalized_length = 0;
    if (normalize_nfc(input, input_length, &normalized,
                      &normalized_length) != 0) return -1;
    size_t position = 0;
    while (position < normalized_length) {
        size_t end;
        if (next_piece(normalized, normalized_length, position, &end) != 0 ||
            end <= position ||
            append_bpe_piece(tokenizer, normalized + position,
                             end - position, output) != 0) {
            free(normalized);
            return -1;
        }
        position = end;
    }
    free(normalized);
    return 0;
}

static const unsigned char *token_bytes(const qwen36_tokenizer *tokenizer,
                                        uint32_t id, size_t *length) {
    const qwen36_token_directory_entry *entry = &tokenizer->directory[id];
    *length = entry->length;
    return tokenizer->token_blob + entry->offset;
}

static size_t find_bytes(const unsigned char *haystack, size_t haystack_length,
                         const unsigned char *needle, size_t needle_length) {
    if (needle_length == 0 || needle_length > haystack_length) return SIZE_MAX;
    size_t limit = haystack_length - needle_length;
    for (size_t index = 0; index <= limit; ++index) {
        if (haystack[index] == needle[0] &&
            memcmp(haystack + index, needle, needle_length) == 0)
            return index;
    }
    return SIZE_MAX;
}

static int find_next_added(const qwen36_tokenizer *tokenizer,
                           const unsigned char *input, size_t input_length,
                           size_t *offset, uint32_t *token_id,
                           size_t *token_length) {
    size_t best_offset = SIZE_MAX;
    size_t best_length = 0;
    uint32_t best_id = 0;
    for (uint32_t id = tokenizer->header->added_first;
         id < tokenizer->header->added_first + tokenizer->header->added_count;
         ++id) {
        size_t length;
        const unsigned char *bytes = token_bytes(tokenizer, id, &length);
        size_t candidate = find_bytes(input, input_length, bytes, length);
        if (candidate < best_offset ||
            (candidate == best_offset && length > best_length)) {
            best_offset = candidate;
            best_length = length;
            best_id = id;
        }
    }
    if (best_offset == SIZE_MAX) return 0;
    *offset = best_offset;
    *token_id = best_id;
    *token_length = best_length;
    return 1;
}

int qwen36_tokenizer_encode(
    const qwen36_tokenizer *tokenizer, const char *utf8,
    uint32_t *token_ids, size_t token_capacity, size_t *token_count,
    char *error_message, size_t error_capacity) {
    if (tokenizer == NULL || utf8 == NULL || token_count == NULL) {
        tokenizer_error(error_message, error_capacity,
                        "invalid tokenizer encode arguments");
        return 1;
    }
    token_vector output = {0};
    const unsigned char *input = (const unsigned char *)utf8;
    size_t remaining = strlen(utf8);
    while (remaining != 0) {
        size_t offset = 0;
        size_t added_length = 0;
        uint32_t added_id = 0;
        int found = find_next_added(tokenizer, input, remaining, &offset,
                                    &added_id, &added_length);
        size_t regular_length = found ? offset : remaining;
        if ((regular_length != 0 &&
             encode_regular(tokenizer, input, regular_length,
                                           &output) != 0) ||
            (found && vector_append(&output, added_id) != 0)) {
            free(output.ids);
            tokenizer_error(error_message, error_capacity,
                            "cannot encode UTF-8 input");
            return 2;
        }
        if (!found) break;
        input += offset + added_length;
        remaining -= offset + added_length;
    }
    *token_count = output.count;
    if (output.count > token_capacity ||
        (output.count != 0 && token_ids == NULL)) {
        free(output.ids);
        tokenizer_error(error_message, error_capacity,
                        "token output requires %zu entries", *token_count);
        return 3;
    }
    memcpy(token_ids, output.ids, output.count * sizeof(*token_ids));
    free(output.ids);
    return 0;
}

int qwen36_tokenizer_decode(
    const qwen36_tokenizer *tokenizer, const uint32_t *token_ids,
    size_t token_count, char *utf8, size_t utf8_capacity,
    size_t *utf8_length, char *error_message, size_t error_capacity) {
    if (tokenizer == NULL || utf8_length == NULL ||
        (token_count != 0 && token_ids == NULL)) {
        tokenizer_error(error_message, error_capacity,
                        "invalid tokenizer decode arguments");
        return 1;
    }
    size_t required = 0;
    for (size_t index = 0; index < token_count; ++index) {
        uint32_t id = token_ids[index];
        if (id >= tokenizer->header->vocab_size ||
            tokenizer->directory[id].length > SIZE_MAX - required) {
            tokenizer_error(error_message, error_capacity,
                            "token ID %u cannot be decoded", id);
            return 2;
        }
        required += tokenizer->directory[id].length;
    }
    *utf8_length = required;
    if (utf8 == NULL || utf8_capacity <= required) {
        tokenizer_error(error_message, error_capacity,
                        "text output requires %zu bytes plus terminator",
                        required);
        return 3;
    }
    size_t written = 0;
    for (size_t index = 0; index < token_count; ++index) {
        size_t length;
        const unsigned char *bytes =
            token_bytes(tokenizer, token_ids[index], &length);
        memcpy(utf8 + written, bytes, length);
        written += length;
    }
    utf8[written] = '\0';
    return 0;
}
