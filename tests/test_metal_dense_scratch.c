/* GPU regression for the double-buffered direct-RHS scratch allocation.
 * Dyadic inputs/weights keep the CPU oracle exact. No model is required. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
bool ds4_log_is_tty(FILE *f) {(void)f;return false;}
static void ck(int ok){if(!ok){fprintf(stderr,"DENSE_SCRATCH_FAIL\n");exit(2);}}
static float weight(uint32_t type,int row,int k){
    int q=(row*3+k*7+(k/32)*5)%16;
    return (type==2?(q-8):q)*0.0625f;
}
int main(void){
    const int K=512,M=128;const size_t bytes=1024*1024;
    unsigned char *map=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);ck(map!=MAP_FAILED);
    ck(ds4_gpu_init());ck(ds4_gpu_set_model_map(map,bytes));
    const uint32_t types[]={2,12,8}; /* GGUF Q4_0, Q4_K, Q8_0. */
    const int rows[]={1,32,64,128,256,2048};int failures=0;
    for(int ti=0;ti<3;ti++){
        uint32_t type=types[ti];int block=type==12?256:32,block_bytes=type==12?144:type==8?34:18;
        memset(map,0,bytes);
        for(int m=0;m<M;m++)for(int b=0;b<K/block;b++){
            unsigned char *p=map+(m*(K/block)+b)*block_bytes;uint16_t scale=0x2c00;memcpy(p,&scale,2); /* 1/16 */
            if(type==12){memset(p+4,1,8);memset(p+12,0x11,4);}
            for(int k=0;k<block;k++){
                int q=(m*3+(b*block+k)*7+((b*block+k)/32)*5)%16;
                if(type==8)p[2+k]=(uint8_t)q;
                else if(type==2)p[2+k%16]|=(uint8_t)(q<<(k/16*4));
                else p[16+(k/64)*32+k%32]|=(uint8_t)(q<<((k/32)%2*4));
            }
        }
        for(int ni=0;ni<6;ni++){
            int N=rows[ni];if(N==1&&type!=8)continue;
            float *x=malloc((size_t)N*K*4),*y=malloc((size_t)N*M*4);ck(x&&y);
            for(int n=0;n<N;n++)for(int k=0;k<K;k++)x[n*K+k]=((n*13+k*3)%17-8)*0.0625f;
            ds4_gpu_tensor *tx=ds4_gpu_tensor_alloc((size_t)N*K*4),*ty=ds4_gpu_tensor_alloc((size_t)N*M*4);ck(tx&&ty);
            ck(ds4_gpu_tensor_write(tx,0,x,(size_t)N*K*4));
            ck(type==8&&N==1?ds4_gpu_matmul_q8_0_decode_mpp_tensor(ty,map,bytes,0,K,M,tx,N):
               ds4_gpu_matmul_quant_tensor(ty,map,bytes,0,type,K,M,tx,N));
            ck(ds4_gpu_synchronize());ck(ds4_gpu_tensor_read(ty,0,y,(size_t)N*M*4));
            double max=0;int bad=0;
            for(int n=0;n<N;n++)for(int m=0;m<M;m++){
                double expected=0;for(int k=0;k<K;k++)expected+=(double)x[n*K+k]*weight(type,m,k);
                double d=fabs(y[n*M+m]-expected);if(d>max)max=d;if(!isfinite(y[n*M+m])||d>0.0001)bad++;
            }
            printf("DENSE_SCRATCH type=%u rows=%d max_error=%.9g bad=%d\n",type,N,max,bad);fflush(stdout);failures+=bad;
            ds4_gpu_tensor_free(tx);ds4_gpu_tensor_free(ty);free(x);free(y);
        }
    }
    /* Registered mmap is released at process exit after the Metal runtime. */
    printf("DENSE_SCRATCH_COMPLETE failures=%d\n",failures);return failures?1:0;
}
