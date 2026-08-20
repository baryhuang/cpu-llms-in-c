// whisper_batch_pipeline.c — CPU/GPU pipelined batch transcription for jetson-orin.
//
// The per-file CLI leaves the six Cortex-A78AE cores idle whenever the GPU runs and
// vice versa, and reloads the model for every file. This driver keeps the model
// resident and runs N worker pipelines over one file queue: while one file's GPU
// stages execute, another file's CPU stages (WAV decode, NEON/FP16 mel FFT, beam
// bookkeeping and sampling between GPU steps) proceed on the idle cores.
//
// Each worker owns a full whisper context (safe isolation; quantized models are
// small enough to be resident twice). Decoding parameters match the e2e driver
// (beam 5, language en, no timestamps) so transcripts are gateable against the
// single-file baseline.
//
// Usage: whisper_batch_pipeline MODEL FILELIST NWORKERS THREADS_PER_WORKER
// Output: JSON on stdout {wall_s, rtfx_wall, per-file hyps}.

#include "whisper.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_FILES 256

typedef struct {
    char  path[512];
    float * pcm;
    int     n_samples;
    char  * text;
    double  t_file;
} job_t;

static job_t g_jobs[MAX_FILES];
static int   g_njobs = 0;
static atomic_int g_next;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

// minimal RIFF reader for the bench set: PCM16 mono 16 kHz
static int load_wav(const char * path, float ** out, int * n_out) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open failed: %s\n", path); return 1; }
    uint8_t hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
        fprintf(stderr, "not a RIFF/WAVE: %s\n", path); fclose(f); return 1;
    }
    uint16_t fmt = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    int16_t * data = NULL;
    uint32_t  data_len = 0;
    for (;;) {
        uint8_t ch[8];
        if (fread(ch, 1, 8, f) != 8) break;
        uint32_t sz;
        memcpy(&sz, ch + 4, 4);
        if (!memcmp(ch, "fmt ", 4)) {
            uint8_t buf[16];
            if (sz < 16 || fread(buf, 1, 16, f) != 16) { fclose(f); return 1; }
            memcpy(&fmt,      buf +  0, 2);
            memcpy(&channels, buf +  2, 2);
            memcpy(&rate,     buf +  4, 4);
            memcpy(&bits,     buf + 14, 2);
            if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
        } else if (!memcmp(ch, "data", 4)) {
            data = malloc(sz);
            if (!data || fread(data, 1, sz, f) != sz) { free(data); fclose(f); return 1; }
            data_len = sz;
        } else {
            fseek(f, (long) sz + (sz & 1), SEEK_CUR);
        }
        if (data) break;
    }
    fclose(f);
    if (!data || fmt != 1 || channels != 1 || rate != 16000 || bits != 16) {
        fprintf(stderr, "expect PCM16 mono 16kHz: %s (fmt=%u ch=%u rate=%u bits=%u)\n",
                path, fmt, channels, rate, bits);
        free(data);
        return 1;
    }
    const int n = (int) (data_len / 2);
    float * pcm = malloc(sizeof(float) * (size_t) n);
    for (int i = 0; i < n; i++) pcm[i] = (float) data[i] / 32768.0f;
    free(data);
    *out = pcm;
    *n_out = n;
    return 0;
}

typedef struct {
    const char * model;
    int          threads;
    int          worker_id;
} worker_arg_t;

static void * worker_main(void * p) {
    worker_arg_t * wa = p;

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;
    struct whisper_context * ctx = whisper_init_from_file_with_params(wa->model, cparams);
    if (!ctx) { fprintf(stderr, "worker %d: model load failed\n", wa->worker_id); exit(1); }

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    params.beam_search.beam_size = 5;
    params.language        = "en";
    params.n_threads       = wa->threads;
    params.no_timestamps   = true;
    params.print_progress  = false;
    params.print_realtime  = false;
    params.print_special   = false;
    params.print_timestamps = false;

    for (;;) {
        const int i = atomic_fetch_add(&g_next, 1);
        if (i >= g_njobs) break;
        job_t * j = &g_jobs[i];
        const double t0 = now_s();
        if (whisper_full(ctx, params, j->pcm, j->n_samples) != 0) {
            fprintf(stderr, "worker %d: whisper_full failed on %s\n", wa->worker_id, j->path);
            exit(1);
        }
        size_t len = 1;
        const int n_seg = whisper_full_n_segments(ctx);
        for (int s = 0; s < n_seg; s++) len += strlen(whisper_full_get_segment_text(ctx, s));
        char * text = calloc(1, len + 1);
        for (int s = 0; s < n_seg; s++) strcat(text, whisper_full_get_segment_text(ctx, s));
        j->text   = text;
        j->t_file = now_s() - t0;
    }

    whisper_free(ctx);
    return NULL;
}

static void json_escape(FILE * f, const char * s) {
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') fputc('\\', f);
        if (*s == '\n') { fputs("\\n", f); continue; }
        fputc(*s, f);
    }
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL FILELIST NWORKERS THREADS_PER_WORKER\n", argv[0]);
        return 1;
    }
    const char * model    = argv[1];
    const int    nworkers = atoi(argv[3]);
    const int    threads  = atoi(argv[4]);

    FILE * fl = fopen(argv[2], "r");
    if (!fl) { fprintf(stderr, "open failed: %s\n", argv[2]); return 1; }
    char line[512];
    double total_audio = 0.0;
    while (fgets(line, sizeof(line), fl) && g_njobs < MAX_FILES) {
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;
        job_t * j = &g_jobs[g_njobs];
        snprintf(j->path, sizeof(j->path), "%s", line);
        if (load_wav(j->path, &j->pcm, &j->n_samples)) return 1;
        total_audio += (double) j->n_samples / 16000.0;
        g_njobs++;
    }
    fclose(fl);

    atomic_store(&g_next, 0);
    pthread_t    th[16];
    worker_arg_t wa[16];

    const double t_start = now_s();  // includes per-worker model load (reported separately via wall_incl_load)
    for (int w = 0; w < nworkers; w++) {
        wa[w] = (worker_arg_t) { model, threads, w };
        if (pthread_create(&th[w], NULL, worker_main, &wa[w])) { fprintf(stderr, "pthread_create failed\n"); return 1; }
    }
    for (int w = 0; w < nworkers; w++) pthread_join(th[w], NULL);
    const double wall = now_s() - t_start;

    printf("{\n \"model\": \"%s\",\n \"nworkers\": %d,\n \"threads_per_worker\": %d,\n \"files\": %d,\n",
           model, nworkers, threads, g_njobs);
    printf(" \"audio_s\": %.1f,\n \"wall_incl_load_s\": %.2f,\n \"rtfx_wall_incl_load\": %.3f,\n \"per_file\": [\n",
           total_audio, wall, total_audio / wall);
    for (int i = 0; i < g_njobs; i++) {
        printf("  {\"file\": \"%s\", \"proc_s\": %.3f, \"hyp\": \"", g_jobs[i].path, g_jobs[i].t_file);
        json_escape(stdout, g_jobs[i].text ? g_jobs[i].text : "");
        printf("\"}%s\n", i + 1 < g_njobs ? "," : "");
    }
    printf(" ]\n}\n");
    return 0;
}
