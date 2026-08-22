#define _POSIX_C_SOURCE 200809L

#include "minimindo_audio_encoder.h"
#include "minimindo_mimi.h"
#include "minimindo_talker.h"
#include "minimindo_thinker.h"
#include "minimindo_tokenizer.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

typedef struct { float score; uint32_t id; } candidate;

static minimindo_thinker *resident_thinker;
static minimindo_talker *resident_talker;
static minimindo_tokenizer *resident_tokenizer;
static minimindo_mimi *resident_mimi;
static minimindo_audio_encoder *resident_audio_encoder;
static const uint32_t resident_context = 512;

static double monotonic_seconds(void);

static int ensure_resident(const char *thinker_path,const char *talker_path,
                           const char *tokenizer_path,const char *mimi_path,
                           const char *audio_encoder_path,char *error,size_t error_capacity)
{
    if(!resident_thinker)resident_thinker=minimindo_thinker_open(thinker_path,resident_context,error,error_capacity);
    if(!resident_talker)resident_talker=minimindo_talker_open(talker_path,resident_context,error,error_capacity);
    if(!resident_tokenizer)resident_tokenizer=minimindo_tokenizer_open(tokenizer_path,error,error_capacity);
    if(!resident_mimi)resident_mimi=minimindo_mimi_open(mimi_path,512,error,error_capacity);
    if(audio_encoder_path&&!resident_audio_encoder)resident_audio_encoder=minimindo_audio_encoder_open(audio_encoder_path,error,error_capacity);
    return resident_thinker&&resident_talker&&resident_tokenizer&&resident_mimi&&
           (!audio_encoder_path||resident_audio_encoder)?0:-1;
}

static uint64_t random_state;
static uint64_t random_u64(void)
{
    uint64_t x = random_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    random_state = x;
    return x * UINT64_C(2685821657736338717);
}
static double random_unit(void) { return (random_u64() >> 11) * 0x1.0p-53; }

static int descending(const void *left, const void *right)
{
    const candidate *a = left, *b = right;
    return a->score < b->score ? 1 : a->score > b->score ? -1 : 0;
}

static uint32_t sample_top_p(const float *logits, uint32_t count,
                             float temperature, float top_p,
                             const uint32_t *history, size_t history_count,
                             float repetition_penalty, candidate *work)
{
    float maximum = -INFINITY;
    for (uint32_t i = 0; i < count; ++i) {
        float score = logits[i];
        if (repetition_penalty != 1.0f)
            for (size_t j = 0; j < history_count; ++j)
                if (history[j] == i) { score = score > 0 ? score / repetition_penalty : score * repetition_penalty; break; }
        work[i] = (candidate){score / temperature, i};
        if (work[i].score > maximum) maximum = work[i].score;
    }
    qsort(work, count, sizeof(*work), descending);
    double denominator = 0;
    for (uint32_t i = 0; i < count; ++i) denominator += exp(work[i].score - maximum);
    double cumulative = 0; uint32_t kept = count;
    for (uint32_t i = 0; i < count; ++i) {
        cumulative += exp(work[i].score - maximum) / denominator;
        if (cumulative >= top_p) { kept = i + 1; break; }
    }
    double selected_total = 0;
    for (uint32_t i = 0; i < kept; ++i) selected_total += exp(work[i].score - maximum);
    const double target = random_unit() * selected_total;
    double running = 0;
    for (uint32_t i = 0; i < kept; ++i) {
        running += exp(work[i].score - maximum);
        if (running >= target) return work[i].id;
    }
    return work[kept - 1].id;
}

static uint32_t sample_top_k(const float *logits, uint32_t count, uint32_t top_k,
                             float temperature, const uint32_t *history,
                             size_t history_count, float repetition_penalty,
                             candidate *work)
{
    for (uint32_t i = 0; i < count; ++i) {
        float score = logits[i];
        for (size_t j = 0; j < history_count; ++j)
            if (history[j] == i) { score = score > 0 ? score / repetition_penalty : score * repetition_penalty; break; }
        work[i] = (candidate){score / temperature, i};
    }
    qsort(work, count, sizeof(*work), descending);
    if (top_k > count) top_k = count;
    const float maximum = work[0].score;
    double total = 0;
    for (uint32_t i = 0; i < top_k; ++i) total += exp(work[i].score - maximum);
    const double target = random_unit() * total;
    double running = 0;
    for (uint32_t i = 0; i < top_k; ++i) {
        running += exp(work[i].score - maximum);
        if (running >= target) return work[i].id;
    }
    return work[top_k - 1].id;
}

static char *format_prompt(const char *user)
{
    static const char prefix[] = "<|im_start|>user\n";
    static const char suffix[] = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const size_t bytes = sizeof(prefix) + strlen(user) + sizeof(suffix);
    char *text = malloc(bytes);
    if (text != NULL) snprintf(text, bytes, "%s%s%s", prefix, user, suffix);
    return text;
}

static int encode(const minimindo_tokenizer *tokenizer, const char *text,
                  uint32_t **ids, size_t *count, char *error, size_t error_capacity)
{
    size_t required = 0;
    (void)minimindo_tokenizer_encode(tokenizer, text, NULL, 0, &required, error, error_capacity);
    *ids = malloc(required * sizeof(**ids));
    if (*ids == NULL) return -1;
    if (minimindo_tokenizer_encode(tokenizer, text, *ids, required, count,
                                   error, error_capacity) != 0) { free(*ids); *ids = NULL; return -1; }
    return 0;
}

static char *decode_text(const minimindo_tokenizer *tokenizer,
                         const uint32_t *ids, size_t count,
                         char *error, size_t error_capacity)
{
    size_t required = 0;
    (void)minimindo_tokenizer_decode(tokenizer, ids, count, NULL, 0, &required, error, error_capacity);
    char *text = malloc(required + 1);
    if (text == NULL) return NULL;
    if (minimindo_tokenizer_decode(tokenizer, ids, count, text, required + 1,
                                   &required, error, error_capacity) != 0) { free(text); return NULL; }
    return text;
}

static void u16(FILE *stream, uint16_t value)
{ unsigned char b[2] = {(unsigned char)value,(unsigned char)(value>>8)}; fwrite(b,1,2,stream); }
static void u32(FILE *stream, uint32_t value)
{ unsigned char b[4] = {(unsigned char)value,(unsigned char)(value>>8),(unsigned char)(value>>16),(unsigned char)(value>>24)}; fwrite(b,1,4,stream); }
static int write_wav(const char *path, const float *audio, size_t samples, uint32_t rate)
{
    FILE *f = fopen(path, "wb"); if (!f || samples > UINT32_MAX / 2) return -1;
    uint32_t bytes = (uint32_t)samples * 2;
    fwrite("RIFF",1,4,f); u32(f,36+bytes); fwrite("WAVEfmt ",1,8,f); u32(f,16);
    u16(f,1); u16(f,1); u32(f,rate); u32(f,rate*2); u16(f,2); u16(f,16);
    fwrite("data",1,4,f); u32(f,bytes);
    for (size_t i=0;i<samples;++i) { float x=audio[i]; if(x>1)x=1;if(x< -1)x=-1;u16(f,(uint16_t)(int16_t)lrintf(x*32767)); }
    int failed=ferror(f); fclose(f); return failed ? -1 : 0;
}

typedef struct {
    minimindo_mimi *model;
    minimindo_mimi_stream *stream;
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    uint32_t *codes;
    float *audio;
    size_t capacity_frames;
    size_t queued_frames;
    size_t decoded_frames;
    size_t samples;
    int drain_threads;
    int producer_done;
    int failed;
    char error[256];
} mimi_decode_worker;

static void *mimi_decode_thread(void *opaque)
{
    mimi_decode_worker *worker = opaque;
    const size_t samples_per_frame =
        minimindo_mimi_samples_for_frames(worker->model, 1);
    while (1) {
        uint32_t frame[MINIMINDO_MIMI_CODEBOOKS];
        pthread_mutex_lock(&worker->mutex);
        while (worker->decoded_frames == worker->queued_frames &&
               !worker->producer_done && !worker->failed)
            pthread_cond_wait(&worker->ready, &worker->mutex);
        if (worker->failed ||
            (worker->producer_done &&
             worker->decoded_frames == worker->queued_frames)) {
            pthread_mutex_unlock(&worker->mutex);
            break;
        }
        const size_t index = worker->decoded_frames;
        for (uint32_t codebook = 0;
             codebook < MINIMINDO_MIMI_CODEBOOKS; ++codebook)
            frame[codebook] =
                worker->codes[index * MINIMINDO_MIMI_CODEBOOKS + codebook];
        const int producer_done = worker->producer_done;
        pthread_mutex_unlock(&worker->mutex);
#if defined(_OPENMP)
        /* Leave the main model nearly all cores while tokens are arriving. */
        omp_set_num_threads(producer_done ? worker->drain_threads : 1);
#else
        (void)producer_done;
#endif
        size_t produced = 0;
        const int result = minimindo_mimi_stream_decode(
            worker->stream, frame, 1,
            worker->audio + index * samples_per_frame, samples_per_frame,
            &produced, worker->error, sizeof(worker->error));
        pthread_mutex_lock(&worker->mutex);
        if (result != 0 || produced != samples_per_frame) {
            worker->failed = 1;
        } else {
            ++worker->decoded_frames;
            worker->samples += produced;
        }
        pthread_cond_broadcast(&worker->ready);
        pthread_mutex_unlock(&worker->mutex);
    }
    return NULL;
}

static int mimi_decode_worker_start(mimi_decode_worker *worker,
                                    minimindo_mimi *model,
                                    size_t capacity_frames)
{
    memset(worker, 0, sizeof(*worker));
    worker->model = model;
    worker->capacity_frames = capacity_frames;
#if defined(_OPENMP)
    worker->drain_threads = omp_get_max_threads();
#else
    worker->drain_threads = 1;
#endif
    const size_t capacity_samples =
        minimindo_mimi_samples_for_frames(model, capacity_frames);
    worker->codes = calloc(capacity_frames * MINIMINDO_MIMI_CODEBOOKS,
                           sizeof(*worker->codes));
    worker->audio = malloc(capacity_samples * sizeof(*worker->audio));
    worker->stream = minimindo_mimi_stream_open(
        model, worker->error, sizeof(worker->error));
    if (worker->codes == NULL || worker->audio == NULL ||
        worker->stream == NULL) goto fail;
    if (pthread_mutex_init(&worker->mutex, NULL) != 0) goto fail;
    if (pthread_cond_init(&worker->ready, NULL) != 0) {
        pthread_mutex_destroy(&worker->mutex);
        goto fail;
    }
    if (pthread_create(&worker->thread, NULL, mimi_decode_thread, worker) != 0) {
        pthread_cond_destroy(&worker->ready);
        pthread_mutex_destroy(&worker->mutex);
        goto fail;
    }
    return 0;
fail:
    minimindo_mimi_stream_close(worker->stream);
    worker->stream = NULL;
    free(worker->codes);
    worker->codes = NULL;
    free(worker->audio);
    worker->audio = NULL;
    return -1;
}

static int mimi_decode_worker_push(mimi_decode_worker *worker,
                                   const uint32_t frame[MINIMINDO_MIMI_CODEBOOKS])
{
    pthread_mutex_lock(&worker->mutex);
    if (worker->failed || worker->queued_frames >= worker->capacity_frames) {
        pthread_mutex_unlock(&worker->mutex);
        return -1;
    }
    for (uint32_t codebook = 0;
         codebook < MINIMINDO_MIMI_CODEBOOKS; ++codebook)
        worker->codes[worker->queued_frames * MINIMINDO_MIMI_CODEBOOKS +
                      codebook] = frame[codebook];
    ++worker->queued_frames;
    pthread_cond_signal(&worker->ready);
    pthread_mutex_unlock(&worker->mutex);
    return 0;
}

static int mimi_decode_worker_finish(mimi_decode_worker *worker)
{
    pthread_mutex_lock(&worker->mutex);
    worker->producer_done = 1;
    pthread_cond_broadcast(&worker->ready);
    pthread_mutex_unlock(&worker->mutex);
    pthread_join(worker->thread, NULL);
    pthread_cond_destroy(&worker->ready);
    pthread_mutex_destroy(&worker->mutex);
    minimindo_mimi_stream_close(worker->stream);
    worker->stream = NULL;
    free(worker->codes);
    worker->codes = NULL;
    return worker->failed ? -1 : 0;
}

static uint16_t read_u16(const unsigned char *p){return(uint16_t)(p[0]|p[1]<<8);}
static uint32_t read_u32(const unsigned char *p){return(uint32_t)(p[0]|p[1]<<8|p[2]<<16|p[3]<<24);}
static int load_wav(const char *path,int16_t **samples,size_t *count)
{
    FILE *f=fopen(path,"rb");if(!f)return -1;unsigned char riff[12];
    if(fread(riff,1,12,f)!=12||memcmp(riff,"RIFF",4)||memcmp(riff+8,"WAVE",4)){fclose(f);return -1;}
    uint16_t format=0,channels=0,bits=0;uint32_t rate=0,bytes=0;unsigned char *data=NULL;
    while(!feof(f)){unsigned char h[8];if(fread(h,1,8,f)!=8)break;uint32_t size=read_u32(h+4);
        if(!memcmp(h,"fmt ",4)){unsigned char v[40];if(size>sizeof(v)||fread(v,1,size,f)!=size)break;format=read_u16(v);channels=read_u16(v+2);rate=read_u32(v+4);bits=read_u16(v+14);}
        else if(!memcmp(h,"data",4)){data=malloc(size);if(!data||fread(data,1,size,f)!=size)break;bytes=size;}
        else fseek(f,size,SEEK_CUR);
        if(size&1)fseek(f,1,SEEK_CUR);}
    fclose(f);if(format!=1||channels!=1||rate!=16000||bits!=16||!data){free(data);return -1;}
    *samples=(int16_t *)data;*count=bytes/2;return 0;
}

static char *audio_prompt(size_t frames)
{
    static const char marker[]="<|audio_pad|>";size_t width=sizeof(marker)-1;
    char *text=malloc(frames*width+1);if(!text)return NULL;
    for(size_t i=0;i<frames;++i)
        memcpy(text+i*width,marker,width);
    text[frames*width]='\0';
    return text;
}

static int run(const char *thinker_path, const char *talker_path,
               const char *tokenizer_path, const char *mimi_path,
               const char *prompt_text, const char *wav_path,
               const char *audio_encoder_path,const char *audio_path,
               const int16_t *provided_pcm,size_t provided_pcm_count,
               uint32_t max_tokens, uint64_t seed, int stream_mimi)
{
    char error[256] = {0};
    const double run_start = monotonic_seconds();
    if(ensure_resident(thinker_path,talker_path,tokenizer_path,mimi_path,audio_encoder_path,error,sizeof(error))){fprintf(stderr,"%s\n",error);return -1;}
    minimindo_thinker *thinker=resident_thinker;minimindo_talker *talker=resident_talker;
    minimindo_tokenizer *tokenizer=resident_tokenizer;minimindo_mimi *mimi=resident_mimi;
    minimindo_thinker_reset(thinker);minimindo_talker_reset(talker);
    const uint32_t tv = minimindo_thinker_vocab_size(thinker), av = minimindo_talker_vocab_size(talker);
    const uint32_t hidden = minimindo_thinker_hidden_size(thinker), pad = minimindo_talker_pad_token(talker);
    float *text_logits = malloc((size_t)tv*sizeof(float));
    float *audio_logits = malloc((size_t)8*av*sizeof(float)); float *bridge=malloc((size_t)hidden*sizeof(float));
    candidate *work=malloc((tv>av?tv:av)*sizeof(candidate));
    uint32_t *generated=calloc(max_tokens,sizeof(uint32_t));
    uint32_t *all_codes=malloc((size_t)8*max_tokens*sizeof(uint32_t));
    uint32_t *frames=malloc((size_t)8*max_tokens*sizeof(uint32_t));
    minimindo_audio_encoder *audio_encoder=resident_audio_encoder;float *audio_embeddings=NULL;size_t audio_frames=0;char *audio_user=NULL;int16_t *owned_pcm=NULL;
    const double audio_encode_start = monotonic_seconds();
    if(audio_path||provided_pcm){const int16_t *pcm=provided_pcm;size_t pcm_count=provided_pcm_count;if(audio_path&&load_wav(audio_path,&owned_pcm,&pcm_count)){fprintf(stderr,"audio input must be 16 kHz mono PCM WAV\n");return -1;}if(owned_pcm)pcm=owned_pcm;
        size_t frame_capacity=minimindo_audio_encoder_frames(pcm_count);audio_embeddings=malloc(frame_capacity*768*sizeof(float));
        if(!audio_encoder||!audio_embeddings||minimindo_audio_encoder_encode_pcm16(audio_encoder,pcm,pcm_count,audio_embeddings,frame_capacity*768,&audio_frames,error,sizeof(error))){fprintf(stderr,"audio encoder: %s\n",error);free(owned_pcm);return -1;}free(owned_pcm);audio_user=audio_prompt(audio_frames);prompt_text=audio_user;}
    const double audio_encode_end = monotonic_seconds();
    char *formatted=format_prompt(prompt_text); uint32_t *prompt=NULL; size_t prompt_count=0;
    if(!text_logits||!audio_logits||!bridge||!work||!generated||!all_codes||!frames||!formatted||
       encode(tokenizer,formatted,&prompt,&prompt_count,error,sizeof(error))!=0) { fprintf(stderr,"allocation/tokenizer: %s\n",error); return -1; }
    uint32_t current_audio[8]; for(int i=0;i<8;++i) current_audio[i]=pad;
    size_t audio_cursor=0;
    for(size_t p=0;p<prompt_count;++p) {
        int thinker_result;
        if(prompt[p]==16&&audio_cursor<audio_frames)thinker_result=minimindo_thinker_forward_embedding(thinker,audio_embeddings+audio_cursor++*768,768,text_logits,tv,bridge,hidden,error,sizeof(error));
        else thinker_result=minimindo_thinker_forward_bridge(thinker,prompt[p],text_logits,tv,bridge,hidden,error,sizeof(error));
        if(thinker_result||
           minimindo_talker_forward(talker,bridge,hidden,current_audio,NULL,0,audio_logits,(size_t)8*av,error,sizeof(error))) { fprintf(stderr,"prefill: %s\n",error);return -1; }
    }
    if(audio_cursor!=audio_frames){fprintf(stderr,"audio token/embedding count mismatch: %zu/%zu\n",audio_cursor,audio_frames);return -1;}
    const double prefill_end = monotonic_seconds();
    mimi_decode_worker decoder;
    int decoder_started = 0;
    if (stream_mimi) {
        if (mimi_decode_worker_start(&decoder, mimi, max_tokens) != 0) {
            fprintf(stderr, "Mimi stream worker: %s\n", decoder.error);
            return -1;
        }
        decoder_started = 1;
    }
    random_state=seed?seed:UINT64_C(1); size_t steps=0, frame_count=0; int text_finished=0, first_finished=1;
    int stop[8]; for(int i=0;i<8;++i)stop[i]=-1;
    while(steps<max_tokens) {
        uint32_t text_token;
        if(text_finished) text_token=first_finished?201:0;
        else text_token=sample_top_p(text_logits,tv,0.75f,0.90f,generated,steps,1.0f,work);
        first_finished=0; generated[steps]=text_token;
        const int audio_step=(int)steps-1;
        for(uint32_t c=0;c<8;++c) {
            uint32_t code=pad;
            if(audio_step>=(int)c) {
                uint32_t recent[3]; size_t recent_count=0;
                for(size_t back=1;back<=3&&back<=steps;++back) recent[recent_count++]=all_codes[(size_t)c*max_tokens+steps-back];
                code=sample_top_k(audio_logits+(size_t)c*av,av,50,0.2f,recent,recent_count,1.05f,work);
                if(stop[c]<0&&code>=2048)stop[c]=(int)steps;
            }
            all_codes[(size_t)c*max_tokens+steps]=code;
        }
        if(audio_step>=7) {
            int active=1; uint32_t frame[8];
            for(uint32_t c=0;c<8;++c) {
                size_t index=steps-7+c; frame[c]=all_codes[(size_t)c*max_tokens+index];
                if(frame[c]>=2048||(stop[c]>=0&&(int)index>=stop[c]))active=0;
            }
            if(active) {
                for(uint32_t c=0;c<8;++c)
                    frames[(size_t)c*max_tokens+frame_count]=frame[c];
                if (decoder_started && mimi_decode_worker_push(&decoder, frame) != 0) {
                    fprintf(stderr, "Mimi stream queue failed\n");
                    (void)mimi_decode_worker_finish(&decoder);
                    free(decoder.audio);
                    return -1;
                }
                ++frame_count;
            }
        }
        ++steps;
        int all_stopped=1;for(int c=0;c<8;++c)if(stop[c]<0)all_stopped=0;
        if(text_finished&&all_stopped)break;
        for(uint32_t c=0;c<8;++c)current_audio[c]=pad;
        for(int c=0;c<8&&c<audio_step+1;++c)current_audio[c]=all_codes[(size_t)c*max_tokens+steps-1];
        if(minimindo_thinker_forward_bridge(thinker,text_token,text_logits,tv,bridge,hidden,error,sizeof(error))||
           minimindo_talker_forward(talker,bridge,hidden,current_audio,NULL,0,audio_logits,(size_t)8*av,error,sizeof(error))) {
            fprintf(stderr,"decode: %s\n",error);
            if (decoder_started) {
                (void)mimi_decode_worker_finish(&decoder);
                free(decoder.audio);
            }
            return -1;
        }
        if(!text_finished&&text_token==2)text_finished=1;
    }
    const double generation_end = monotonic_seconds();
    size_t text_count=steps; while(text_count&&generated[text_count-1]==0) --text_count;
    char *answer=decode_text(tokenizer,generated,text_count,error,sizeof(error));
    if(frame_count==0){
        fprintf(stderr,"Talker produced no complete Mimi frames\n");
        if (decoder_started) {
            (void)mimi_decode_worker_finish(&decoder);
            free(decoder.audio);
        }
        return -1;
    }
    uint32_t *mimi_codes = NULL;
    float *audio = NULL;
    size_t samples = 0;
    if (decoder_started) {
        if (mimi_decode_worker_finish(&decoder) != 0) {
            fprintf(stderr, "Mimi stream decode: %s\n", decoder.error);
            free(decoder.audio);
            return -1;
        }
        audio = decoder.audio;
        samples = decoder.samples;
    } else {
        mimi_codes=malloc((size_t)8*frame_count*sizeof(uint32_t));
        if(!mimi_codes){fprintf(stderr,"Mimi code compaction allocation failed\n");return -1;}
        for(uint32_t c=0;c<8;++c)for(size_t t=0;t<frame_count;++t)
            mimi_codes[(size_t)c*frame_count+t]=frames[(size_t)c*max_tokens+t];
        const size_t sample_capacity=minimindo_mimi_samples_for_frames(mimi,frame_count);
        audio=malloc(sample_capacity*sizeof(float));
        if(!audio||minimindo_mimi_decode(mimi,mimi_codes,frame_count,audio,sample_capacity,&samples,error,sizeof(error))){
            fprintf(stderr,"Mimi decode: %s\n",error);
            free(audio);
            free(mimi_codes);
            return -1;
        }
    }
    const double mimi_end = monotonic_seconds();
    if(write_wav(wav_path,audio,samples,minimindo_mimi_sample_rate(mimi))){
        fprintf(stderr,"Mimi/WAV write failed\n");
        free(audio);
        free(mimi_codes);
        return -1;
    }
    printf("{\"text\":\"");
    for(const unsigned char *p=(unsigned char *)(answer?answer:"");*p;++p){if(*p=='"'||*p=='\\')putchar('\\');if(*p=='\n'){fputs("\\n",stdout);continue;}putchar(*p);}
    printf("\",\"steps\":%zu,\"audio_frames\":%zu,\"samples\":%zu,"
           "\"seed\":%llu,\"audio_encode_ms\":%.0f,\"prefill_ms\":%.0f,"
           "\"generate_ms\":%.0f,\"mimi_drain_ms\":%.0f,"
           "\"model_ms\":%.0f,\"stream_mimi\":%s}\n",
           steps,frame_count,samples,
           (unsigned long long)seed,
           (audio_encode_end-audio_encode_start)*1000,
           (prefill_end-audio_encode_end)*1000,
           (generation_end-prefill_end)*1000,
           (mimi_end-generation_end)*1000,
           (mimi_end-run_start)*1000,
           decoder_started?"true":"false");
    free(audio);free(mimi_codes);free(answer);free(prompt);free(formatted);free(audio_user);free(audio_embeddings);free(text_logits);free(audio_logits);free(bridge);free(work);free(generated);free(all_codes);free(frames);
    return 0;
}

static int warm_resident(const char *thinker,const char *talker,const char *tokenizer,
                         const char *mimi,const char *audio_encoder)
{
    char error[256]={0};if(ensure_resident(thinker,talker,tokenizer,mimi,audio_encoder,error,sizeof(error))){fprintf(stderr,"warm-up open: %s\n",error);return -1;}
    uint32_t tv=minimindo_thinker_vocab_size(resident_thinker),av=minimindo_talker_vocab_size(resident_talker),h=minimindo_thinker_hidden_size(resident_thinker);
    float *tl=malloc((size_t)tv*4),*al=malloc((size_t)8*av*4),*bridge=malloc((size_t)h*4);uint32_t ids[8];for(int i=0;i<8;++i)ids[i]=minimindo_talker_pad_token(resident_talker);
    if(!tl||!al||!bridge||minimindo_thinker_forward_bridge(resident_thinker,1,tl,tv,bridge,h,error,sizeof(error))||minimindo_talker_forward(resident_talker,bridge,h,ids,NULL,0,al,(size_t)8*av,error,sizeof(error)))return -1;
    if(audio_encoder){int16_t silence[800]={0};float embedding[768*2];size_t frames=0;if(minimindo_audio_encoder_encode_pcm16(resident_audio_encoder,silence,800,embedding,768*2,&frames,error,sizeof(error)))return -1;}
    uint32_t codes[8]={0};float audio[1920];size_t samples=0;if(minimindo_mimi_decode(resident_mimi,codes,1,audio,1920,&samples,error,sizeof(error)))return -1;
    free(tl);free(al);free(bridge);minimindo_thinker_reset(resident_thinker);minimindo_talker_reset(resident_talker);return 0;
}

static double monotonic_seconds(void)
{struct timespec value;clock_gettime(CLOCK_MONOTONIC,&value);return value.tv_sec+value.tv_nsec*1e-9;}

static int safe_alsa_name(const char *name)
{if(!name||!*name)return 0;for(const unsigned char *p=(const unsigned char *)name;*p;++p)if(!((*p>='a'&&*p<='z')||(*p>='A'&&*p<='Z')||(*p>='0'&&*p<='9')||strchr(":,._-",*p)))return 0;return 1;}

static FILE *open_capture(const char *device)
{
    char command[256];if(!safe_alsa_name(device))return NULL;
    snprintf(command,sizeof(command),"arecord -q -D %s -t raw -f S16_LE -c 1 -r 16000",device);
    return popen(command,"r");
}

enum { CAPTURE_CHUNK=512, CAPTURE_QUEUE=64 };
typedef struct {
    FILE *stream;
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    int16_t chunks[CAPTURE_QUEUE][CAPTURE_CHUNK];
    size_t read_index,write_index,count;
    int stopped,failed;
} capture_queue;

static void *capture_worker(void *opaque)
{
    capture_queue *queue=opaque;int16_t chunk[CAPTURE_CHUNK];
    while(!queue->stopped){size_t got=fread(chunk,sizeof(int16_t),CAPTURE_CHUNK,queue->stream);
        if(got!=CAPTURE_CHUNK){if(feof(queue->stream)){clearerr(queue->stream);continue;}queue->failed=1;break;}
        pthread_mutex_lock(&queue->mutex);
        if(queue->count==CAPTURE_QUEUE){queue->read_index=(queue->read_index+1)%CAPTURE_QUEUE;--queue->count;}
        memcpy(queue->chunks[queue->write_index],chunk,sizeof(chunk));queue->write_index=(queue->write_index+1)%CAPTURE_QUEUE;++queue->count;
        pthread_cond_signal(&queue->ready);pthread_mutex_unlock(&queue->mutex);}
    return NULL;
}

static int capture_start(capture_queue *queue,const char *device)
{
    memset(queue,0,sizeof(*queue));queue->stream=open_capture(device);if(!queue->stream)return -1;
    if(pthread_mutex_init(&queue->mutex,NULL)||pthread_cond_init(&queue->ready,NULL)||pthread_create(&queue->thread,NULL,capture_worker,queue))return -1;
    return 0;
}

static int capture_next(capture_queue *queue,int16_t output[CAPTURE_CHUNK])
{
    pthread_mutex_lock(&queue->mutex);while(!queue->count&&!queue->failed)pthread_cond_wait(&queue->ready,&queue->mutex);
    if(queue->failed){pthread_mutex_unlock(&queue->mutex);return -1;}
    memcpy(output,queue->chunks[queue->read_index],CAPTURE_CHUNK*sizeof(int16_t));queue->read_index=(queue->read_index+1)%CAPTURE_QUEUE;--queue->count;
    pthread_mutex_unlock(&queue->mutex);return 0;
}

static void capture_flush(capture_queue *queue)
{pthread_mutex_lock(&queue->mutex);queue->read_index=queue->write_index;queue->count=0;pthread_mutex_unlock(&queue->mutex);}

static void capture_stop(capture_queue *queue)
{
    queue->stopped=1;pclose(queue->stream);pthread_join(queue->thread,NULL);
    pthread_cond_destroy(&queue->ready);pthread_mutex_destroy(&queue->mutex);
}

static int live(const char *thinker,const char *talker,const char *tokenizer,
                const char *mimi,const char *audio_encoder,const char *capture_device,
                const char *playback_device,uint32_t max_tokens,uint64_t seed)
{
    if(!safe_alsa_name(capture_device)||!safe_alsa_name(playback_device))return -1;
    const double warm_start=monotonic_seconds();
    if(warm_resident(thinker,talker,tokenizer,mimi,audio_encoder))return -1;
    printf("READY pipeline=MiniMind-O-native-C warmup_ms=%.0f capture=%s playback=%s\n",
           (monotonic_seconds()-warm_start)*1000,capture_device,playback_device);fflush(stdout);
    capture_queue capture;if(capture_start(&capture,capture_device)){perror("arecord");return -1;}
    enum{CHUNK=CAPTURE_CHUNK,PREROLL=8192,MAX_SPEECH=16*16000};
    int16_t chunk[CHUNK],ring[PREROLL],*speech=malloc(MAX_SPEECH*sizeof(int16_t));
    size_t ring_write=0,ring_count=0,speech_count=0;double noise=120.0,last_monitor=0;int speaking=0,hot=0,silent=0,active_chunks=0,cooldown=0;unsigned turn=0;
    if(!speech)return -1;
    while(1){if(capture_next(&capture,chunk))break;size_t got=CHUNK;
        double squares=0,peak=0;for(size_t i=0;i<got;++i){double v=chunk[i];squares+=v*v;if(fabs(v)>peak)peak=fabs(v);}double rms=sqrt(squares/got);
        double threshold=fmax(280.0,noise*3.2);double now=monotonic_seconds();
        if(now-last_monitor>=0.5){printf("%s level_rms=%.0f peak=%.0f noise=%.0f threshold=%.0f turn=%u\n",speaking?"SPEECH":"LISTEN",rms,peak,noise,threshold,turn);fflush(stdout);last_monitor=now;}
        if(cooldown){--cooldown;if(rms<threshold)noise=noise*.98+rms*.02;continue;}
        if(!speaking){if(rms<threshold)noise=noise*.995+rms*.005;
            for(size_t i=0;i<got;++i){ring[ring_write]=chunk[i];ring_write=(ring_write+1)%PREROLL;if(ring_count<PREROLL)++ring_count;}
            hot=rms>threshold?hot+1:0;if(hot>=2){speaking=1;silent=0;active_chunks=hot;speech_count=0;size_t start=(ring_write+PREROLL-ring_count)%PREROLL;
                for(size_t i=0;i<ring_count&&speech_count<MAX_SPEECH;++i)speech[speech_count++]=ring[(start+i)%PREROLL];
                printf("EVENT speech_start turn=%u preroll_ms=%zu\n",turn+1,ring_count*1000/16000);fflush(stdout);}}
        else{if(speech_count+got<=MAX_SPEECH){memcpy(speech+speech_count,chunk,got*sizeof(int16_t));speech_count+=got;}if(rms>threshold)++active_chunks;
            silent=rms<threshold*.65?silent+1:0;
            if(silent>=20||speech_count+got>MAX_SPEECH){speaking=0;hot=0;ring_count=0;++turn;size_t trim=(size_t)silent*CHUNK;if(trim<speech_count)speech_count-=trim;
                if(speech_count<4000||active_chunks<3){printf("EVENT empty_vad turn=%u samples=%zu active_chunks=%d action=discard\n",turn,speech_count,active_chunks);fflush(stdout);speech_count=0;continue;}
                printf("EVENT speech_end turn=%u samples=%zu duration_ms=%zu\n",turn,speech_count,speech_count*1000/16000);fflush(stdout);
                const double inference_start=monotonic_seconds();const char *response="/dev/shm/minimindo-live-response.wav";
                int result=run(thinker,talker,tokenizer,mimi,NULL,response,audio_encoder,NULL,speech,speech_count,max_tokens,seed+turn,1);
                printf("EVENT inference_end turn=%u elapsed_ms=%.0f result=%d\n",turn,(monotonic_seconds()-inference_start)*1000,result);fflush(stdout);speech_count=0;
                if(!result){char command[320];snprintf(command,sizeof(command),"aplay -q -D %s %s",playback_device,response);int playback=system(command);printf("EVENT playback_end turn=%u result=%d\n",turn,playback);fflush(stdout);}capture_flush(&capture);cooldown=32;
            }}
    }
    free(speech);capture_stop(&capture);return -1;
}

int main(int argc,char **argv)
{
    if(argc<8){fprintf(stderr,"usage: %s THINKER TALKER TOKENIZER MIMI (--prompt TEXT | --audio INPUT.wav --audio-encoder ENCODER.mmo) --output FILE.wav [--stream-mimi] [--max-tokens N] [--seed N]\n",argv[0]);return 2;}
    const char *prompt=NULL,*output=NULL,*audio_path=NULL,*audio_encoder=NULL,*capture=NULL,*playback=NULL;int live_mode=0,stream_mimi=0;uint32_t max_tokens=96;uint64_t seed=1;
    for(int i=5;i<argc;++i){if(!strcmp(argv[i],"--prompt")&&i+1<argc)prompt=argv[++i];else if(!strcmp(argv[i],"--audio")&&i+1<argc)audio_path=argv[++i];else if(!strcmp(argv[i],"--audio-encoder")&&i+1<argc)audio_encoder=argv[++i];else if(!strcmp(argv[i],"--output")&&i+1<argc)output=argv[++i];else if(!strcmp(argv[i],"--live"))live_mode=1;else if(!strcmp(argv[i],"--stream-mimi"))stream_mimi=1;else if(!strcmp(argv[i],"--capture-device")&&i+1<argc)capture=argv[++i];else if(!strcmp(argv[i],"--playback-device")&&i+1<argc)playback=argv[++i];else if(!strcmp(argv[i],"--max-tokens")&&i+1<argc)max_tokens=(uint32_t)strtoul(argv[++i],NULL,10);else if(!strcmp(argv[i],"--seed")&&i+1<argc)seed=strtoull(argv[++i],NULL,10);else return 2;}
    if(max_tokens<16)return 2;
    if(live_mode)return live(argv[1],argv[2],argv[3],argv[4],audio_encoder,capture?capture:"plughw:1,0",playback?playback:"plughw:0,0",max_tokens,seed)!=0;
    if((!prompt&&!audio_path)||(prompt&&audio_path)||!output)return 2;
    return run(argv[1],argv[2],argv[3],argv[4],prompt,output,audio_encoder,audio_path,NULL,0,max_tokens,seed,stream_mimi)!=0;
}
