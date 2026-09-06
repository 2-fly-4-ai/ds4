#include "ds4_gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
bool ds4_log_is_tty(FILE *f) { (void)f; return false; }
static void ck(int ok) { if (!ok) { fprintf(stderr,"INDEXER_TEST_FAIL\n"); exit(2); } }
static unsigned seed = 931;
static float rnd(void) { seed=seed*1664525u+1013904223u; return ((int)(seed%20001)-10000)/7319.0f; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(void) {
    ck(ds4_gpu_init());
    const unsigned sizes[]={512,513,514,532,767,768,769,1023,1024,1025,2048,8192};
    float q[64*128],w[64];
    for(unsigned si=0;si<sizeof(sizes)/sizeof(sizes[0]);si++) {
        unsigned n=sizes[si];
        float *k=malloc(n*128*4),*a=malloc(n*4),*b=malloc(n*4);
        ck(k&&a&&b);
        ds4_gpu_tensor *tq=ds4_gpu_tensor_alloc(sizeof(q)), *tw=ds4_gpu_tensor_alloc(sizeof(w));
        ds4_gpu_tensor *tk=ds4_gpu_tensor_alloc(n*128*4), *ts=ds4_gpu_tensor_alloc(n*4);
        ds4_gpu_tensor *ti=ds4_gpu_tensor_alloc(512*4);
        ck(tq&&tw&&tk&&ts&&ti);
        for(int trial=0;trial<8;trial++) {
            for(unsigned i=0;i<64*128;i++)q[i]=trial==0?0:rnd();
            for(unsigned i=0;i<64;i++)w[i]=rnd();
            for(unsigned i=0;i<n*128;i++)k[i]=trial==1?1:rnd();
            ck(ds4_gpu_tensor_write(tq,0,q,sizeof(q))); ck(ds4_gpu_tensor_write(tw,0,w,sizeof(w)));
            ck(ds4_gpu_tensor_write(tk,0,k,n*128*4));
            unsigned ids[2][512];
            for(int mode=0;mode<2;mode++) {
                if(mode)unsetenv("DS4_METAL_DISABLE_INDEXER_STAGED_SCORE"); else setenv("DS4_METAL_DISABLE_INDEXER_STAGED_SCORE","1",1);
                if(mode)unsetenv("DS4_METAL_DISABLE_SHORT_TOPK_SHUFFLE"); else setenv("DS4_METAL_DISABLE_SHORT_TOPK_SHUFFLE","1",1);
                ck(ds4_gpu_indexer_score_one_tensor(ts,tq,tw,tk,n,64,128,1/sqrtf(8192)));
                ck(ds4_gpu_synchronize()); ck(ds4_gpu_tensor_read(ts,0,mode?b:a,n*4));
                ck(ds4_gpu_indexer_topk_tensor(ti,ts,n,1,512));
                ck(ds4_gpu_synchronize()); ck(ds4_gpu_tensor_read(ti,0,ids[mode],512*4));
            }
            unsigned different=0;for(unsigned i=0;i<n;i++)different+=memcmp(a+i,b+i,4)!=0;
            printf("INDEXER_PARITY n=%u trial=%d score_diffs=%u ids_equal=%d\n",n,trial,different,!memcmp(ids[0],ids[1],512*4));
            ck(!different&&!memcmp(ids[0],ids[1],512*4));
        }
        for(int stage=0;stage<2;stage++)for(int mode=0;mode<2;mode++) {
            if(mode)unsetenv("DS4_METAL_DISABLE_INDEXER_STAGED_SCORE");else setenv("DS4_METAL_DISABLE_INDEXER_STAGED_SCORE","1",1);
            if(mode)unsetenv("DS4_METAL_DISABLE_SHORT_TOPK_SHUFFLE");else setenv("DS4_METAL_DISABLE_SHORT_TOPK_SHUFFLE","1",1);
            double t=now();ck(ds4_gpu_begin_commands());
            for(int i=0;i<128;i++) {
                if(!stage)ck(ds4_gpu_indexer_score_one_tensor(ts,tq,tw,tk,n,64,128,1/sqrtf(8192)));
                else ck(ds4_gpu_indexer_topk_tensor(ti,ts,n,1,512));
            }
            ck(ds4_gpu_end_commands());
            printf("INDEXER_TIMING n=%u stage=%s mode=%d us=%.3f\n",n,stage?"topk":"score",mode,(now()-t)*1e6/128);
        }
        ds4_gpu_tensor_free(tq);ds4_gpu_tensor_free(tw);ds4_gpu_tensor_free(tk);ds4_gpu_tensor_free(ts);ds4_gpu_tensor_free(ti);
        free(k);free(a);free(b);fflush(stdout);
    }
    puts("INDEXER_TEST_PASS");return 0;
}
