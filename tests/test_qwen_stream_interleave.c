#include "../ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int ok, const char *why) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", why); exit(1); }
}
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    /* 80 slots deliberately force cross-layer eviction on every token. */
    ds4_engine_options opt = {.model_path=argv[1], .ple_path=argv[2],
        .backend=DS4_BACKEND_METAL, .context_size=1024,
        .ssd_streaming=true, .ssd_streaming_cache_experts=80};
    ds4_engine *e=NULL;
    check(ds4_engine_open(&e,&opt)==0,"engine");
    ds4_session *ref=NULL,*a=NULL,*b=NULL;
    check(ds4_session_create(&ref,e,1024)==0,"reference");
    check(ds4_session_create(&a,e,1024)==0,"A");
    check(ds4_session_create(&b,e,1024)==0,"B");
    ds4_tokens prompt={0}; char err[512]={0};
    ds4_tokenize_text(e,"Explain how a bounded cache evicts old experts.",&prompt);
    check(ds4_session_sync(ref,&prompt,err,sizeof(err))==0,err);
    ds4_session_snapshot snapshot={0};
    check(ds4_session_save_snapshot(ref,&snapshot,err,sizeof(err))==0,err);
    const int V=ds4_engine_vocab_size(e), steps=8;
    float *expected=malloc((size_t)steps*V*sizeof(float));
    float *actual=malloc((size_t)V*sizeof(float)); int tokens[8];
    check(expected && actual,"logit allocation");
    for(int i=0;i<steps;i++) {
        tokens[i]=ds4_session_argmax(ref);
        check(ds4_session_eval(ref,tokens[i],err,sizeof(err))==0,err);
        check(ds4_session_copy_logits(ref,expected+(size_t)i*V,V)==V,"reference logits");
    }
    check(ds4_session_load_snapshot(a,&snapshot,err,sizeof(err))==0,err);
    for(int i=0;i<steps;i++) {
        char text[2048]; int used=snprintf(text,sizeof(text),"Other session %d. ",i);
        for(int j=0;j<20;j++) used+=snprintf(text+used,sizeof(text)-(size_t)used,
            "Independent prefill changes the active layer mapping and evicts experts. ");
        ds4_tokens p={0}; ds4_tokenize_text(e,text,&p);
        check(ds4_session_sync(b,&p,err,sizeof(err))==0,err);
        ds4_tokens_free(&p);
        check(ds4_session_eval(a,tokens[i],err,sizeof(err))==0,err);
        check(ds4_session_copy_logits(a,actual,V)==V,"A logits");
        check(memcmp(actual,expected+(size_t)i*V,(size_t)V*sizeof(float))==0,
              "snapshot/interleave/eviction logits mismatch");
    }
    puts("PASS: eight full-logit steps after snapshot restore and interleaved prefills, 80-slot cache");
    ds4_session_snapshot_free(&snapshot);
    free(actual); free(expected); ds4_tokens_free(&prompt);
    ds4_session_free(b); ds4_session_free(a); ds4_session_free(ref); ds4_engine_close(e);
    return 0;
}
