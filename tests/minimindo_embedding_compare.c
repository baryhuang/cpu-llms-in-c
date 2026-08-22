#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float *read_f32(const char *path,size_t *count)
{
    FILE *file=fopen(path,"rb");
    if(!file||fseek(file,0,SEEK_END))return NULL;
    long bytes=ftell(file);
    if(bytes<=0||bytes%(long)sizeof(float)||fseek(file,0,SEEK_SET)){
        fclose(file);return NULL;
    }
    float *values=malloc((size_t)bytes);
    if(!values||fread(values,1,(size_t)bytes,file)!=(size_t)bytes){
        free(values);fclose(file);return NULL;
    }
    fclose(file);*count=(size_t)bytes/sizeof(float);return values;
}

int main(int argc,char **argv)
{
    if(argc!=3){fprintf(stderr,"usage: %s REFERENCE.f32 TEST.f32\n",argv[0]);return 2;}
    size_t reference_count=0,test_count=0;
    float *reference=read_f32(argv[1],&reference_count);
    float *test=read_f32(argv[2],&test_count);
    if(!reference||!test||reference_count!=test_count){
        fprintf(stderr,"incompatible embedding files\n");return 3;
    }
    double reference_sq=0,test_sq=0,dot=0,error_sq=0,maximum=0;
    for(size_t i=0;i<reference_count;++i){
        double a=reference[i],b=test[i],error=fabs(a-b);
        reference_sq+=a*a;test_sq+=b*b;dot+=a*b;error_sq+=error*error;
        if(error>maximum)maximum=error;
    }
    printf("{\"values\":%zu,\"cosine\":%.9f,\"rmse\":%.9f,"
           "\"max_abs_error\":%.9f}\n",reference_count,
           dot/sqrt(reference_sq*test_sq),sqrt(error_sq/reference_count),
           maximum);
    free(reference);free(test);return 0;
}
