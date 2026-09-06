/* Pool state regression: completed speculative groups must retain the raw
 * rows needed when only a prefix is accepted. No model checkpoint needed. */
#include "ds4_gpu.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
bool ds4_log_is_tty(FILE *f) { (void)f; return false; }
static void ck(int ok) { if (!ok) { fprintf(stderr,"POOL_TEST_ERROR\n"); exit(2); } }
enum { D=128, CAP=32, BYTES=CAP*D*4 };
typedef struct { ds4_gpu_tensor *cache,*key,*gate; } state;
static state alloc_state(void) {
    state s={ds4_gpu_tensor_alloc(BYTES),ds4_gpu_tensor_alloc(4*D*4),ds4_gpu_tensor_alloc(4*D*4)};
    ck(s.cache&&s.key&&s.gate);
    ck(ds4_gpu_tensor_fill_f32(s.cache,0,BYTES/4));
    ck(ds4_gpu_tensor_fill_f32(s.key,0,4*D)); ck(ds4_gpu_tensor_fill_f32(s.gate,0,4*D));
    return s;
}
static void release(state s) { ds4_gpu_tensor_free(s.cache);ds4_gpu_tensor_free(s.key);ds4_gpu_tensor_free(s.gate); }
static void update(state s,void *map,ds4_gpu_tensor *k,ds4_gpu_tensor *g,int pos,int n,bool half) {
    ds4_gpu_tensor *kv=ds4_gpu_tensor_view(k,pos*D*4,n*D*4),*gv=ds4_gpu_tensor_view(g,pos*D*4,n*D*4);
    ck(kv&&gv);
    ck(ds4_gpu_glm53_indexer_pool_update_tensor(s.cache,s.key,s.gate,kv,gv,
        map,1048576,0,D*4,D*8,pos,n,CAP,D,4,1e-6f,half));
    ds4_gpu_tensor_free(kv);ds4_gpu_tensor_free(gv);
}
static bool equal(ds4_gpu_tensor *a,ds4_gpu_tensor *b,size_t n) {
    if (!n) return true;
    unsigned char x[BYTES],y[BYTES];ck(n<=BYTES);
    ck(ds4_gpu_tensor_read(a,0,x,n));ck(ds4_gpu_tensor_read(b,0,y,n));return !memcmp(x,y,n);
}
int main(void) {
    void *map=mmap(NULL,1048576,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);ck(map!=MAP_FAILED);
    for(int d=0;d<D;d++)((float *)map)[d]=1;
    ck(ds4_gpu_init());ck(ds4_gpu_set_model_map(map,1048576));
    float raw[CAP*D],gate[CAP*D];
    for(int i=0;i<CAP*D;i++){raw[i]=((i*17+i/D*13)%97-48)*.03125f;gate[i]=((i*7+i/D)%29-14)*.0625f;}
    ds4_gpu_tensor *k=ds4_gpu_tensor_alloc(BYTES),*g=ds4_gpu_tensor_alloc(BYTES);ck(k&&g);
    ck(ds4_gpu_tensor_write(k,0,raw,BYTES));ck(ds4_gpu_tensor_write(g,0,gate,BYTES));
    int failures=0,cases=0;
    for(int half=0;half<2;half++)for(int pos=0;pos<8;pos++)for(int n=1;n<=8;n*=2){
        /* Multi-group rollback needs snapshots; this kernel's contract is
         * the final group. Cover all full accepts and two-token rejection. */
        for(int reject=0;reject<= (n==2);reject++){
            state a=alloc_state(),b=alloc_state();
            for(int t=0;t<pos;t++){update(a,map,k,g,t,1,half);update(b,map,k,g,t,1,half);}
            update(a,map,k,g,pos,n,half);
            int end=pos+(reject?1:n);
            for(int t=pos;t<end;t++)update(b,map,k,g,t,1,half);
            ck(ds4_gpu_synchronize());
            bool ok=equal(a.cache,b.cache,(end/4)*D*(half?2:4))&&
                equal(a.key,b.key,(end%4)*D*4)&&equal(a.gate,b.gate,(end%4)*D*4);
            cases++;if(!ok){failures++;printf("POOL_FAIL half=%d pos=%d n=%d reject=%d\n",half,pos,n,reject);}
            release(a);release(b);
        }
    }
    /* Lookup can reject across multiple pools, unlike the native two-row
     * case above. Preserve only the live pre-block tail, then replay the
     * accepted prefix. This covers every acceptance length through 16. */
    ds4_gpu_tensor *backup=ds4_gpu_tensor_alloc(8*D*4);ck(backup!=NULL);
    for(int half=0;half<2;half++)for(int pos=0;pos<8;pos++)for(int n=4;n<=16;n*=2){
        for(int accepted=1;accepted<=n;accepted++){
            state a=alloc_state(),b=alloc_state();
            for(int t=0;t<pos;t++){update(a,map,k,g,t,1,half);update(b,map,k,g,t,1,half);}
            size_t tail=(pos%4)*D*4;
            if(tail){ck(ds4_gpu_begin_commands());ck(ds4_gpu_tensor_copy(backup,0,a.key,0,tail));ck(ds4_gpu_tensor_copy(backup,4*D*4,a.gate,0,tail));ck(ds4_gpu_end_commands());}
            update(a,map,k,g,pos,n,half);
            if(accepted<n){
                if(tail){ck(ds4_gpu_begin_commands());ck(ds4_gpu_tensor_copy(a.key,0,backup,0,tail));ck(ds4_gpu_tensor_copy(a.gate,0,backup,4*D*4,tail));ck(ds4_gpu_end_commands());}
                for(int t=pos;t<pos+accepted;t++)update(a,map,k,g,t,1,half);
            }
            int end=pos+accepted;
            for(int t=pos;t<end;t++)update(b,map,k,g,t,1,half);
            ck(ds4_gpu_synchronize());
            bool ok=equal(a.cache,b.cache,(end/4)*D*(half?2:4))&&
                equal(a.key,b.key,(end%4)*D*4)&&equal(a.gate,b.gate,(end%4)*D*4);
            cases++;if(!ok){failures++;printf("LOOKUP_POOL_FAIL half=%d pos=%d n=%d accepted=%d\n",half,pos,n,accepted);}
            release(a);release(b);
        }
    }
    ds4_gpu_tensor_free(backup);ds4_gpu_tensor_free(k);ds4_gpu_tensor_free(g);
    printf("POOL_COMPLETE cases=%d failures=%d\n",cases,failures);return failures?1:0;
}
