/* Short GLM verification must reproduce scalar reductions, not merely the
 * same argmax. Non-dyadic activations expose matrix/matvec rounding changes.
 * No checkpoint required. Includes the former N16 attention TensorOps shape. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
bool ds4_log_is_tty(FILE *f) {(void)f;return false;}
static void ck(int ok) {
    if (!ok) {fprintf(stderr,"SHORT_MATMUL_ERROR\n");exit(2);}
}
static uint32_t seed=12345;
static uint32_t random_bits(void) {seed=seed*1664525u+1013904223u;return seed;}
static int run(ds4_gpu_tensor *y,void *map,size_t bytes,int type,int k,int m,
               ds4_gpu_tensor *x,int n) {
    return type==30 ? ds4_gpu_glm53_matmul_bf16(y,map,bytes,0,k,m,x,n) :
                     ds4_gpu_matmul_quant_tensor(y,map,bytes,0,type,k,m,x,n);
}
int main(void) {
    const size_t bytes=64u*1024u*1024u;
    unsigned char *map=mmap(NULL,bytes,PROT_READ|PROT_WRITE,
                            MAP_PRIVATE|MAP_ANON,-1,0);
    ck(map!=MAP_FAILED);ck(ds4_gpu_init());ck(ds4_gpu_set_model_map(map,bytes));
    ds4_gpu_set_glm_model(true);
    const int shapes[][3]={{30,16384,24},{12,4096,128},{12,8192,4096}};
    const int widths[]={1,2,4,8,9,15,16};int failures=0;
    for(size_t s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) {
        int type=shapes[s][0],k=shapes[s][1],m=shapes[s][2];
        if(type==30) {
            for(int i=0;i<k*m;i++) {
                float f=((int)(random_bits()%2001)-1000)/713.0f;
                uint32_t bits;memcpy(&bits,&f,4);uint16_t bf=bits>>16;
                memcpy(map+2*i,&bf,2);
            }
        } else {
            for(int i=0;i<k/256*m;i++) {
                unsigned char *p=map+144*i;
                uint16_t d=0x21a3,dmin=0x1db7;
                memcpy(p,&d,2);memcpy(p+2,&dmin,2);
                for(int j=4;j<144;j++)p[j]=(unsigned char)(random_bits()>>24);
            }
        }
        float *x=malloc(16u*k*4),*y=malloc(16u*m*4),*ref=malloc(16u*m*4);
        ck(x&&y&&ref);
        for(int i=0;i<16*k;i++)x[i]=((int)(random_bits()%20001)-10000)/7319.0f;
        ds4_gpu_tensor *tx=ds4_gpu_tensor_alloc(16u*k*4),*ty=ds4_gpu_tensor_alloc(16u*m*4);
        ck(tx&&ty);ck(ds4_gpu_tensor_write(tx,0,x,16u*k*4));
        for(int row=0;row<16;row++) {
            ds4_gpu_tensor *vx=ds4_gpu_tensor_view(tx,(uint64_t)row*k*4,k*4);
            ck(vx!=NULL);ck(run(ty,map,bytes,type,k,m,vx,1));ck(ds4_gpu_synchronize());
            ck(ds4_gpu_tensor_read(ty,0,ref+row*m,m*4));ds4_gpu_tensor_free(vx);
        }
        for(size_t w=0;w<sizeof(widths)/sizeof(widths[0]);w++) {
            int n=widths[w],bad=0;ck(run(ty,map,bytes,type,k,m,tx,n));
            ck(ds4_gpu_synchronize());ck(ds4_gpu_tensor_read(ty,0,y,(size_t)n*m*4));
            for(int i=0;i<n*m;i++)bad+=!isfinite(y[i])||memcmp(y+i,ref+i,4)!=0;
            printf("SHORT_MATMUL type=%d k=%d m=%d rows=%d different=%d\n",type,k,m,n,bad);
            failures+=bad!=0;
        }
        ds4_gpu_tensor_free(tx);ds4_gpu_tensor_free(ty);free(x);free(y);free(ref);
    }
    printf("SHORT_MATMUL_COMPLETE failures=%d\n",failures);
    return failures?1:0;
}
