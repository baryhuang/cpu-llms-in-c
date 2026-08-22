#include "minimindo_parallel.h"
#include "minimindo_audio_encoder.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t le16(const unsigned char *p){return(uint16_t)(p[0]|p[1]<<8);}
static uint32_t le32(const unsigned char *p){return(uint32_t)(p[0]|p[1]<<8|p[2]<<16|p[3]<<24);}

static int load_wav(const char *path,int16_t **samples,size_t *count)
{
    FILE *f=fopen(path,"rb");if(!f)return -1;unsigned char riff[12];
    if(fread(riff,1,12,f)!=12||memcmp(riff,"RIFF",4)||memcmp(riff+8,"WAVE",4)){fclose(f);return -1;}
    uint16_t format=0,channels=0,bits=0;uint32_t rate=0;unsigned char *data=NULL;uint32_t bytes=0;
    while(!feof(f)){unsigned char h[8];if(fread(h,1,8,f)!=8)break;uint32_t size=le32(h+4);
        if(!memcmp(h,"fmt ",4)){unsigned char v[40];if(size>sizeof(v)||fread(v,1,size,f)!=size)break;format=le16(v);channels=le16(v+2);rate=le32(v+4);bits=le16(v+14);}
        else if(!memcmp(h,"data",4)){data=malloc(size);if(!data||fread(data,1,size,f)!=size)break;bytes=size;}
        else fseek(f,size,SEEK_CUR);if(size&1)fseek(f,1,SEEK_CUR);}
    fclose(f);if(format!=1||channels!=1||rate!=16000||bits!=16||!data){free(data);return -1;}
    *samples=(int16_t *)data;*count=bytes/2;return 0;
}

int main(int argc,char **argv)
{
    if(argc<3||argc>4){fprintf(stderr,"usage: %s AUDIO_ENCODER.mmo INPUT.wav [OUTPUT.f32]\n",argv[0]);return 2;}
    int16_t *samples=NULL;size_t sample_count=0;if(load_wav(argv[2],&samples,&sample_count)){fprintf(stderr,"invalid 16k mono PCM WAV\n");return 3;}
    size_t capacity=minimindo_audio_encoder_frames(sample_count);float *output=malloc(capacity*768*sizeof(float));char error[256]={0};size_t frames=0;
    minimindo_audio_encoder *model=minimindo_audio_encoder_open(argv[1],error,sizeof(error));
    if(!model||!output){fprintf(stderr,"%s\n",error);return 4;}
    if(minimindo_parallel_session_begin(4U)!=0)return 4;
    int encode_result=minimindo_audio_encoder_encode_pcm16(model,samples,sample_count,output,capacity*768,&frames,error,sizeof(error));
    minimindo_parallel_session_end();
    if(encode_result){fprintf(stderr,"%s\n",error);return 4;}
    double sum=0,peak=0;for(size_t i=0;i<frames*768;++i){sum+=(double)output[i]*output[i];if(fabs(output[i])>peak)peak=fabs(output[i]);}
    if(argc==4){FILE *f=fopen(argv[3],"wb");if(!f||fwrite(output,sizeof(float),frames*768,f)!=frames*768)return 5;fclose(f);}
    printf("{\"samples\":%zu,\"frames\":%zu,\"rms\":%.9g,\"peak\":%.9g}\n",sample_count,frames,sqrt(sum/(frames*768)),peak);
    minimindo_audio_encoder_close(model);free(output);free(samples);return 0;
}
