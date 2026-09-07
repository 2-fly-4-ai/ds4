/* Live regression: another session's prefill must not invalidate a hot
 * session's cached Metal model-map assumption. One model, serial execution. */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int ok, const char *message) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    ds4_engine_options opt = {.model_path=argv[1], .backend=DS4_BACKEND_METAL,
        .context_size=1024, .ssd_streaming=true,
        .ssd_streaming_cache_bytes=8ull*1024*1024*1024};
    ds4_engine *e=NULL;
    check(ds4_engine_open(&e, &opt)==0, "engine");
    ds4_session *ref=NULL, *a=NULL, *b=NULL;
    check(ds4_session_create(&ref,e,1024)==0, "reference");
    check(ds4_session_create(&a,e,1024)==0, "session A");
    check(ds4_session_create(&b,e,1024)==0, "session B");
    char err[512]={0};
    ds4_tokens prompt={0};
    ds4_tokenize_text(e,"Explain why an expert cache reduces repeated disk reads.",&prompt);
    check(ds4_session_sync(ref,&prompt,err,sizeof(err))==0,err);
    const int vocab=ds4_engine_vocab_size(e), steps=16;
    float *expected=malloc((size_t)steps*vocab*sizeof(float));
    float *actual=malloc((size_t)vocab*sizeof(float));
    int tokens[16];
    check(expected && actual,"logit buffers");
    for(int i=0;i<steps;i++) {
        tokens[i]=ds4_session_argmax(ref);
        check(ds4_session_eval(ref,tokens[i],err,sizeof(err))==0,err);
        check(ds4_session_copy_logits(ref,expected+(size_t)i*vocab,vocab)==vocab,"reference logits");
    }
    check(ds4_session_sync(a,&prompt,err,sizeof(err))==0,err);
    for(int i=0;i<steps;i++) {
        char other[4096];
        int used=snprintf(other,sizeof(other),"Independent session %d. ",i);
        for(int j=0;j<40;j++) used+=snprintf(other+used,sizeof(other)-(size_t)used,
            "This paragraph must be prefetched independently of the cached conversation. ");
        ds4_tokens p={0}; ds4_tokenize_text(e,other,&p);
        check(p.len>128 && p.len<1024,"prefill length");
        check(ds4_session_sync(b,&p,err,sizeof(err))==0,err);
        ds4_tokens_free(&p);
        check(ds4_session_eval(a,tokens[i],err,sizeof(err))==0,err);
        check(ds4_session_copy_logits(a,actual,vocab)==vocab,"A logits");
        check(memcmp(actual,expected+(size_t)i*vocab,(size_t)vocab*sizeof(float))==0,
              "interleaved full logits differ");
    }
    puts("PASS: 16 interleaved prefills/decode steps, full logits byte-identical");
    free(actual); free(expected); ds4_tokens_free(&prompt);
    ds4_session_free(b); ds4_session_free(a); ds4_session_free(ref); ds4_engine_close(e);
    return 0;
}
