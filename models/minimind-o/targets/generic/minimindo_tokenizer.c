#define _POSIX_C_SOURCE 200809L

#include "minimindo_tokenizer.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum { TOKENIZER_VERSION = 1, TOKENIZER_HEADER_BYTES = 4096 };

typedef struct {
    uint64_t offset;
    uint32_t length;
    uint32_t flags;
} token_entry;

typedef struct {
    uint32_t left, right, result, rank;
} merge_entry;

typedef struct {
    unsigned char magic[8];
    uint32_t version, header_bytes, vocab_size, merge_count, added_count;
    uint32_t token_entry_bytes, merge_entry_bytes, reserved;
    uint64_t directory_offset, directory_bytes, blob_offset, blob_bytes;
    uint64_t merges_offset, file_bytes;
    char source_sha256[64];
    uint32_t byte_token_ids[256];
} tokenizer_header;

struct minimindo_tokenizer {
    int file;
    const unsigned char *mapping;
    size_t mapped_bytes;
    const tokenizer_header *header;
    const token_entry *directory;
    const unsigned char *blob;
    const merge_entry *merges;
};

typedef struct {
    uint32_t *ids;
    size_t count, capacity;
} token_vector;

static void set_error(char *message, size_t capacity, const char *format, ...)
{
    if (message == NULL || capacity == 0) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(message, capacity, format, arguments);
    va_end(arguments);
}

static int range_ok(uint64_t offset, uint64_t length, uint64_t total)
{
    return offset <= total && length <= total - offset;
}

minimindo_tokenizer *minimindo_tokenizer_open(
    const char *image_path, char *error, size_t error_capacity)
{
    minimindo_tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (tokenizer == NULL) return NULL;
    tokenizer->file = -1;
    tokenizer->file = open(image_path, O_RDONLY);
    struct stat status;
    if (tokenizer->file < 0 || fstat(tokenizer->file, &status) != 0 ||
        status.st_size < TOKENIZER_HEADER_BYTES) {
        set_error(error, error_capacity, "cannot open tokenizer: %s",
                  strerror(errno));
        minimindo_tokenizer_close(tokenizer);
        return NULL;
    }
    tokenizer->mapped_bytes = (size_t)status.st_size;
    tokenizer->mapping = mmap(NULL, tokenizer->mapped_bytes, PROT_READ,
                              MAP_PRIVATE, tokenizer->file, 0);
    if (tokenizer->mapping == MAP_FAILED) {
        tokenizer->mapping = NULL;
        set_error(error, error_capacity, "cannot mmap tokenizer: %s",
                  strerror(errno));
        minimindo_tokenizer_close(tokenizer);
        return NULL;
    }
    tokenizer->header = (const tokenizer_header *)tokenizer->mapping;
    static const unsigned char magic[8] = {'M','M','O','T','O','K','1','\0'};
    const tokenizer_header *header = tokenizer->header;
    if (memcmp(header->magic, magic, 8) != 0 ||
        header->version != TOKENIZER_VERSION ||
        header->header_bytes != TOKENIZER_HEADER_BYTES ||
        header->token_entry_bytes != sizeof(token_entry) ||
        header->merge_entry_bytes != sizeof(merge_entry) ||
        header->reserved != 0 || header->file_bytes != tokenizer->mapped_bytes ||
        header->directory_bytes != (uint64_t)header->vocab_size * sizeof(token_entry) ||
        !range_ok(header->directory_offset, header->directory_bytes,
                  tokenizer->mapped_bytes) ||
        !range_ok(header->blob_offset, header->blob_bytes,
                  tokenizer->mapped_bytes) ||
        !range_ok(header->merges_offset,
                  (uint64_t)header->merge_count * sizeof(merge_entry),
                  tokenizer->mapped_bytes)) {
        set_error(error, error_capacity, "invalid tokenizer image");
        minimindo_tokenizer_close(tokenizer);
        return NULL;
    }
    tokenizer->directory = (const token_entry *)(tokenizer->mapping +
                                                  header->directory_offset);
    tokenizer->blob = tokenizer->mapping + header->blob_offset;
    tokenizer->merges = (const merge_entry *)(tokenizer->mapping +
                                               header->merges_offset);
    uint32_t added = 0;
    for (uint32_t id = 0; id < header->vocab_size; ++id) {
        if (!range_ok(tokenizer->directory[id].offset,
                      tokenizer->directory[id].length, header->blob_bytes)) {
            set_error(error, error_capacity, "token %u is out of range", id);
            minimindo_tokenizer_close(tokenizer);
            return NULL;
        }
        added += (tokenizer->directory[id].flags & 1U) != 0;
    }
    if (added != header->added_count) {
        set_error(error, error_capacity, "added-token count mismatch");
        minimindo_tokenizer_close(tokenizer);
        return NULL;
    }
    return tokenizer;
}

void minimindo_tokenizer_close(minimindo_tokenizer *tokenizer)
{
    if (tokenizer == NULL) return;
    if (tokenizer->mapping != NULL)
        munmap((void *)tokenizer->mapping, tokenizer->mapped_bytes);
    if (tokenizer->file >= 0) close(tokenizer->file);
    free(tokenizer);
}

static int reserve(token_vector *vector, size_t requested)
{
    if (requested <= vector->capacity) return 0;
    size_t capacity = vector->capacity == 0 ? 64 : vector->capacity;
    while (capacity < requested) {
        if (capacity > SIZE_MAX / 2) return -1;
        capacity *= 2;
    }
    uint32_t *next = realloc(vector->ids, capacity * sizeof(*next));
    if (next == NULL) return -1;
    vector->ids = next;
    vector->capacity = capacity;
    return 0;
}

static int append(token_vector *vector, uint32_t id)
{
    if (reserve(vector, vector->count + 1) != 0) return -1;
    vector->ids[vector->count++] = id;
    return 0;
}

static const unsigned char *token_bytes(const minimindo_tokenizer *tokenizer,
                                        uint32_t id, size_t *length)
{
    *length = tokenizer->directory[id].length;
    return tokenizer->blob + tokenizer->directory[id].offset;
}

static const merge_entry *find_merge(const minimindo_tokenizer *tokenizer,
                                     uint32_t left, uint32_t right)
{
    const uint64_t key = ((uint64_t)left << 32) | right;
    size_t low = 0, high = tokenizer->header->merge_count;
    while (low < high) {
        const size_t middle = low + (high - low) / 2;
        const merge_entry *entry = &tokenizer->merges[middle];
        const uint64_t candidate = ((uint64_t)entry->left << 32) | entry->right;
        if (candidate < key) low = middle + 1;
        else high = middle;
    }
    if (low == tokenizer->header->merge_count) return NULL;
    const merge_entry *entry = &tokenizer->merges[low];
    return entry->left == left && entry->right == right ? entry : NULL;
}

static int append_bpe(const minimindo_tokenizer *tokenizer,
                      const unsigned char *bytes, size_t length,
                      token_vector *output)
{
    if (length == 0) return 0;
    uint32_t *piece = malloc(length * sizeof(*piece));
    if (piece == NULL) return -1;
    for (size_t index = 0; index < length; ++index)
        piece[index] = tokenizer->header->byte_token_ids[bytes[index]];
    size_t count = length;
    for (;;) {
        uint32_t rank = UINT32_MAX, left = 0, right = 0, result = 0;
        for (size_t index = 0; index + 1 < count; ++index) {
            const merge_entry *merge = find_merge(tokenizer, piece[index],
                                                   piece[index + 1]);
            if (merge != NULL && merge->rank < rank) {
                rank = merge->rank;
                left = merge->left;
                right = merge->right;
                result = merge->result;
            }
        }
        if (rank == UINT32_MAX) break;
        size_t read = 0, write = 0;
        while (read < count) {
            if (read + 1 < count && piece[read] == left &&
                piece[read + 1] == right) {
                piece[write++] = result;
                read += 2;
            } else {
                piece[write++] = piece[read++];
            }
        }
        count = write;
    }
    if (reserve(output, output->count + count) != 0) {
        free(piece);
        return -1;
    }
    memcpy(output->ids + output->count, piece, count * sizeof(*piece));
    output->count += count;
    free(piece);
    return 0;
}

typedef struct {
    uint32_t codepoint;
    size_t end;
    int letter, number, whitespace, punctuation;
} unicode_unit;

static int in_range(uint32_t value, uint32_t first, uint32_t last)
{
    return value >= first && value <= last;
}

static int utf8_unit(const unsigned char *text, size_t length,
                     size_t position, unicode_unit *unit)
{
    if (position >= length) return -1;
    const unsigned char first = text[position++];
    uint32_t value;
    unsigned continuation;
    if (first < 0x80) { value = first; continuation = 0; }
    else if ((first & 0xe0) == 0xc0) { value = first & 0x1f; continuation = 1; }
    else if ((first & 0xf0) == 0xe0) { value = first & 0x0f; continuation = 2; }
    else if ((first & 0xf8) == 0xf0) { value = first & 0x07; continuation = 3; }
    else return -1;
    for (unsigned index = 0; index < continuation; ++index) {
        if (position >= length || (text[position] & 0xc0) != 0x80) return -1;
        value = (value << 6) | (text[position++] & 0x3f);
    }
    unit->codepoint = value;
    unit->end = position;
    unit->whitespace = value == ' ' || in_range(value, 9, 13) ||
                       value == 0x85 || value == 0xa0 || value == 0x1680 ||
                       in_range(value, 0x2000, 0x200a) || value == 0x2028 ||
                       value == 0x2029 || value == 0x202f || value == 0x205f ||
                       value == 0x3000;
    unit->number = in_range(value, '0', '9') ||
                   in_range(value, 0xff10, 0xff19);
    const int ascii_letter = in_range(value, 'A', 'Z') ||
                             in_range(value, 'a', 'z');
    const int unicode_punctuation = in_range(value, 0x2000, 0x206f) ||
        in_range(value, 0x2e00, 0x2e7f) || in_range(value, 0x3001, 0x303f) ||
        in_range(value, 0xff01, 0xff0f) || in_range(value, 0xff1a, 0xff20) ||
        in_range(value, 0xff3b, 0xff40) || in_range(value, 0xff5b, 0xff65);
    unit->punctuation = !unit->whitespace && !unit->number &&
        ((value < 128 && !ascii_letter) || unicode_punctuation);
    unit->letter = !unit->whitespace && !unit->number && !unit->punctuation;
    return 0;
}

static int ascii_fold_match(const unsigned char *text, size_t available,
                            const char *pattern)
{
    const size_t length = strlen(pattern);
    if (length > available) return 0;
    for (size_t index = 0; index < length; ++index) {
        unsigned char value = text[index];
        if (value >= 'A' && value <= 'Z') value += 'a' - 'A';
        if (value != (unsigned char)pattern[index]) return 0;
    }
    return (int)length;
}

static int next_piece(const unsigned char *text, size_t length,
                      size_t start, size_t *end)
{
    static const char *contractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    if (text[start] == '\'') {
        for (size_t index = 0; index < sizeof(contractions) / sizeof(contractions[0]); ++index) {
            const int match = ascii_fold_match(text + start, length - start,
                                               contractions[index]);
            if (match) { *end = start + (size_t)match; return 0; }
        }
    }
    unicode_unit first, current;
    if (utf8_unit(text, length, start, &first) != 0) return -1;
    size_t cursor = start;
    if (first.letter) cursor = first.end;
    else if (first.codepoint == ' ' && first.end < length) {
        unicode_unit next;
        if (utf8_unit(text, length, first.end, &next) == 0 && next.letter)
            cursor = next.end;
    }
    if (cursor != start) {
        while (cursor < length && utf8_unit(text, length, cursor, &current) == 0 &&
               current.letter) cursor = current.end;
        *end = cursor;
        return 0;
    }
    cursor = start;
    if (first.number) cursor = first.end;
    else if (first.codepoint == ' ' && first.end < length) {
        unicode_unit next;
        if (utf8_unit(text, length, first.end, &next) == 0 && next.number)
            cursor = next.end;
    }
    if (cursor != start) {
        while (cursor < length && utf8_unit(text, length, cursor, &current) == 0 &&
               current.number) cursor = current.end;
        *end = cursor;
        return 0;
    }
    cursor = start;
    if (first.punctuation) cursor = first.end;
    else if (first.codepoint == ' ' && first.end < length) {
        unicode_unit next;
        if (utf8_unit(text, length, first.end, &next) == 0 && next.punctuation)
            cursor = next.end;
    }
    if (cursor != start) {
        while (cursor < length && utf8_unit(text, length, cursor, &current) == 0 &&
               current.punctuation) cursor = current.end;
        while (cursor < length && utf8_unit(text, length, cursor, &current) == 0 &&
               (current.codepoint == '\r' || current.codepoint == '\n')) cursor = current.end;
        *end = cursor;
        return 0;
    }
    if (first.whitespace) {
        cursor = start;
        size_t before_last = 0, units = 0;
        while (cursor < length && utf8_unit(text, length, cursor, &current) == 0 &&
               current.whitespace) {
            before_last = cursor;
            cursor = current.end;
            ++units;
        }
        *end = units > 1 && cursor < length ? before_last : cursor;
        return 0;
    }
    *end = first.end;
    return 0;
}

static size_t find_bytes(const unsigned char *haystack, size_t haystack_length,
                         const unsigned char *needle, size_t needle_length)
{
    if (needle_length == 0 || needle_length > haystack_length) return SIZE_MAX;
    for (size_t index = 0; index <= haystack_length - needle_length; ++index)
        if (haystack[index] == needle[0] &&
            memcmp(haystack + index, needle, needle_length) == 0) return index;
    return SIZE_MAX;
}

static int find_special(const minimindo_tokenizer *tokenizer,
                        const unsigned char *input, size_t length,
                        size_t *offset, size_t *token_length, uint32_t *id)
{
    size_t best_offset = SIZE_MAX, best_length = 0;
    uint32_t best_id = 0;
    for (uint32_t candidate = 0; candidate < tokenizer->header->vocab_size; ++candidate) {
        if ((tokenizer->directory[candidate].flags & 1U) == 0) continue;
        size_t size;
        const unsigned char *bytes = token_bytes(tokenizer, candidate, &size);
        const size_t found = find_bytes(input, length, bytes, size);
        if (found < best_offset || (found == best_offset && size > best_length)) {
            best_offset = found;
            best_length = size;
            best_id = candidate;
        }
    }
    if (best_offset == SIZE_MAX) return 0;
    *offset = best_offset;
    *token_length = best_length;
    *id = best_id;
    return 1;
}

static int encode_regular(const minimindo_tokenizer *tokenizer,
                          const unsigned char *text, size_t length,
                          token_vector *output)
{
    size_t position = 0;
    while (position < length) {
        size_t end;
        if (next_piece(text, length, position, &end) != 0 || end <= position ||
            append_bpe(tokenizer, text + position, end - position, output) != 0)
            return -1;
        position = end;
    }
    return 0;
}

int minimindo_tokenizer_encode(
    const minimindo_tokenizer *tokenizer, const char *utf8,
    uint32_t *token_ids, size_t token_capacity, size_t *token_count,
    char *error, size_t error_capacity)
{
    if (tokenizer == NULL || utf8 == NULL || token_count == NULL) return -1;
    token_vector output = {0};
    const unsigned char *input = (const unsigned char *)utf8;
    size_t remaining = strlen(utf8);
    while (remaining != 0) {
        size_t offset = 0, special_length = 0;
        uint32_t special_id = 0;
        const int found = find_special(tokenizer, input, remaining, &offset,
                                       &special_length, &special_id);
        const size_t regular = found ? offset : remaining;
        if ((regular && encode_regular(tokenizer, input, regular, &output) != 0) ||
            (found && append(&output, special_id) != 0)) {
            free(output.ids);
            set_error(error, error_capacity, "cannot encode UTF-8 input");
            return -2;
        }
        if (!found) break;
        input += offset + special_length;
        remaining -= offset + special_length;
    }
    *token_count = output.count;
    if (output.count > token_capacity || (output.count && token_ids == NULL)) {
        free(output.ids);
        set_error(error, error_capacity, "token output requires %zu entries",
                  *token_count);
        return -3;
    }
    memcpy(token_ids, output.ids, output.count * sizeof(*token_ids));
    free(output.ids);
    return 0;
}

int minimindo_tokenizer_decode(
    const minimindo_tokenizer *tokenizer, const uint32_t *token_ids,
    size_t token_count, char *utf8, size_t utf8_capacity,
    size_t *utf8_length, char *error, size_t error_capacity)
{
    if (tokenizer == NULL || utf8_length == NULL ||
        (token_count && token_ids == NULL)) return -1;
    size_t required = 0;
    for (size_t index = 0; index < token_count; ++index) {
        if (token_ids[index] >= tokenizer->header->vocab_size ||
            tokenizer->directory[token_ids[index]].length > SIZE_MAX - required) {
            set_error(error, error_capacity, "invalid token ID");
            return -2;
        }
        required += tokenizer->directory[token_ids[index]].length;
    }
    *utf8_length = required;
    if (utf8 == NULL || utf8_capacity <= required) {
        set_error(error, error_capacity, "text output requires %zu bytes", required + 1);
        return -3;
    }
    size_t written = 0;
    for (size_t index = 0; index < token_count; ++index) {
        size_t length;
        const unsigned char *bytes = token_bytes(tokenizer, token_ids[index], &length);
        memcpy(utf8 + written, bytes, length);
        written += length;
    }
    utf8[written] = '\0';
    return 0;
}
