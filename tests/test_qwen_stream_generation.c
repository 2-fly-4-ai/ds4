#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}
static void check(int ok,const char *s) {
    if(!ok) {fprintf(stderr,"FAIL: %s\n",s);exit(1);}
}
/* Same executable can link against baseline or candidate objects. The binary
 * trace contains each cycle's accepted tokens and the complete final logits. */
int main(int argc,char **argv) {
    if(argc!=8) return 2;
    int budget=atoi(argv[3]),mtp=atoi(argv[4]),cycles=atoi(argv[5]);
    ds4_engine_options opt={.model_path=argv[1],.ple_path=argv[2],
        .backend=DS4_BACKEND_METAL,.context_size=4096,.glm_mtp=mtp!=0,
        .ssd_streaming=budget!=0,.ssd_streaming_cache_bytes=(uint64_t)budget*1024*1024*1024};
    ds4_engine *e=NULL; ds4_session *s=NULL; ds4_tokens p={0}; char err[512]={0};
    check(ds4_engine_open(&e,&opt)==0,"open");
    check(ds4_session_create(&s,e,4096)==0,"session");
    ds4_tokenize_text(e,argv[6],&p);
    double start=now(); check(ds4_session_sync(s,&p,err,sizeof(err))==0,err);
    double prefill=now()-start;
    int V=ds4_engine_vocab_size(e); float *logits=malloc((size_t)V*sizeof(float));
    check(logits!=NULL,"logits");
    FILE *trace=fopen(argv[7],"wb"); check(trace!=NULL,"trace");
    int total=0,doubles=0; start=now();
    for(int i=0;i<cycles;i++) {
        int first=ds4_session_argmax(s),accepted[2]={0};
        int n;
        if(mtp) n=ds4_session_eval_speculative_argmax(s,first,2,-1,accepted,2,err,sizeof(err));
        else {check(ds4_session_eval(s,first,err,sizeof(err))==0,err);accepted[0]=first;n=1;}
        check(n>0 && n<=2,err); total+=n; doubles+=n==2;
        check(ds4_session_copy_logits(s,logits,V)==V,"logits read");
        check(fwrite(&n,sizeof(n),1,trace)==1,"count write");
        check(fwrite(accepted,sizeof(int),(size_t)n,trace)==(size_t)n,"tokens write");
        check(fwrite(logits,sizeof(float),(size_t)V,trace)==(size_t)V,"logits write");
    }
    double decode=now()-start; check(fclose(trace)==0,"trace close");
    printf("cycles=%d tokens=%d doubles=%d prefill_ms=%.3f decode_ms=%.3f tps=%.3f (includes logit tracing)\n",
        cycles,total,doubles,prefill*1000,decode*1000,total/decode);
    free(logits); ds4_tokens_free(&p); ds4_session_free(s); ds4_engine_close(e);
    return 0;
}
