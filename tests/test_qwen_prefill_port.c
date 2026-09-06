#define main original_hot_main
#include "test_ds4_hot_rewind.c"
#undef main

int main(int argc, char **argv) {
    ck(argc == 2, "model");
    const int length = getenv("HOT_CTX") ? atoi(getenv("HOT_CTX")) : 8192;
    ds4_engine_options opt = {.model_path=argv[1], .backend=DS4_BACKEND_METAL,
        .context_size=length+256, .power_percent=100, .warm_weights=true};
    ds4_engine *e = NULL;
    ck(ds4_engine_open(&e, &opt) == 0, "engine");
    ds4_tokens text={0}, prompt={0};
    ds4_tokenize_text(e, "Archive entry: preserve verified records and revision history. Review the supplied code carefully.\n", &text);
    for (int i=0; i<length; i++) token_vec_push(&prompt, text.v[i%text.len]);
    ds4_session_snapshot reference={0}, continuation={0};
    char err[512]={0};
    /* Two excluded warmups, then balanced ABBA/BAAB timed order. */
    const int order[]={0,1,0,1,1,0,1,0,0,1};
    for (int repeat=0; repeat<10; repeat++) {
        if (order[repeat]) unsetenv("DS4_QWEN_NORM_REUSE_DISABLE");
        else setenv("DS4_QWEN_NORM_REUSE_DISABLE", "1", 1);
        ds4_session *s=NULL;
        ck(ds4_session_create(&s,e,opt.context_size)==0,"session");
        double start=now_sec();
        ck(ds4_session_sync(s,&prompt,err,sizeof(err))==0,err);
        double elapsed=now_sec()-start;
        if (!repeat) ck(ds4_session_save_snapshot(s,&reference,err,sizeof(err))==0,err);
        else same_snapshot(s,&reference);
        for (int n=0;n<16;n++) ck(ds4_session_eval(s,ds4_session_argmax(s),err,sizeof(err))==0,err);
        if (!repeat) ck(ds4_session_save_snapshot(s,&continuation,err,sizeof(err))==0,err);
        else same_snapshot(s,&continuation);
        printf("QWEN_PREFILL_%s mode=%d repeat=%d prompt=%d ms=%.4f tps=%.4f exact=1\n",
               repeat<2?"WARMUP":"TIMING",order[repeat],repeat,length,elapsed*1000,length/elapsed);
        fflush(stdout);
        ds4_session_free(s);
    }
    ds4_session_snapshot_free(&reference);
    ds4_session_snapshot_free(&continuation);
    ds4_tokens_free(&prompt); ds4_tokens_free(&text); ds4_engine_close(e);
    puts("QWEN_PREFILL_PORT_PASS");
    return 0;
}
