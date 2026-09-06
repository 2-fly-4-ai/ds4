#define main original_hot_main
#include "test_ds4_hot_rewind.c"
#undef main
int main(int argc, char **argv) {
    ck(argc==2 && getenv("HOT_TASK"), "model and HOT_TASK");
    uint32_t context=getenv("HOT_CTX")?atoi(getenv("HOT_CTX")):2048;
    ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,
        .context_size=context+1024,.power_percent=100,.warm_weights=true,
        .ssd_streaming=getenv("HOT_SSD")!=NULL,.ssd_streaming_cache_bytes=8ull<<30};
    ds4_engine *e=NULL;ck(ds4_engine_open(&e,&opt)==0,"engine");
    ds4_tokens prompt={0};ds4_encode_chat_prompt(e,NULL,getenv("HOT_TASK"),DS4_THINK_NONE,&prompt);
    ck(prompt.len>=(int)context,"prompt long enough");prompt.len=context;
    ds4_session *s=NULL;char error[512]={0};ck(ds4_session_create(&s,e,opt.context_size)==0,"session");
    ck(ds4_session_sync(s,&prompt,error,sizeof(error))==0,error);
    ds4_session_snapshot initial={0},final={0};ck(ds4_session_save_snapshot(s,&initial,error,sizeof(error))==0,error);
    int tokens[128];float *logits=malloc(128ull*DS4_N_VOCAB*4);ck(logits!=NULL,"logits");
    for(int mode=0;mode<2;mode++) {
        if(mode) {ck(ds4_session_load_snapshot(s,&initial,error,sizeof(error))==0,error);unsetenv("DS4_METAL_DISABLE_INDEXER_STAGED_SCORE");}
        else setenv("DS4_METAL_DISABLE_INDEXER_STAGED_SCORE","1",1);
        if(mode)unsetenv("DS4_METAL_DISABLE_SHORT_TOPK_SHUFFLE"); else setenv("DS4_METAL_DISABLE_SHORT_TOPK_SHUFFLE","1",1);
        for(int step=0;step<128;step++) {
            if(!mode)tokens[step]=ds4_session_argmax(s);
            ck(tokens[step]==ds4_session_argmax(s),"argmax parity");
            ck(ds4_session_eval(s,tokens[step],error,sizeof(error))==0,error);
            if(!mode)memcpy(logits+(size_t)step*DS4_N_VOCAB,s->logits,DS4_N_VOCAB*4);
            else ck(!memcmp(logits+(size_t)step*DS4_N_VOCAB,s->logits,DS4_N_VOCAB*4),"full logit parity");
        }
        if(!mode)ck(ds4_session_save_snapshot(s,&final,error,sizeof(error))==0,error);
        else same_snapshot(s,&final);
    }
    printf("INDEXER_LIVE_PARITY_PASS context=%u ssd=%d tokens=128 full_logits=128 snapshot_bytes=%llu\n",context,opt.ssd_streaming,(unsigned long long)final.len);
    if(getenv("HOT_TIMING")) {
        const int order[]={0,1,1,0,1,0,0,1};
        for(int repeat=0;repeat<8;repeat++) {
            int mode=order[repeat];
            ck(ds4_session_load_snapshot(s,&initial,error,sizeof(error))==0,error);
            if(mode) {unsetenv("DS4_METAL_DISABLE_INDEXER_STAGED_SCORE");unsetenv("DS4_METAL_DISABLE_SHORT_TOPK_SHUFFLE");}
            else {setenv("DS4_METAL_DISABLE_INDEXER_STAGED_SCORE","1",1);setenv("DS4_METAL_DISABLE_SHORT_TOPK_SHUFFLE","1",1);}
            double t=now_sec();
            for(int step=0;step<128;step++)ck(ds4_session_eval(s,tokens[step],error,sizeof(error))==0,error);
            double elapsed=now_sec()-t;
            printf("INDEXER_LIVE_TIMING context=%u repeat=%d mode=%d ms=%.4f tps=%.4f\n",context,repeat,mode,elapsed*1000,128/elapsed);fflush(stdout);
        }
    }
    free(logits);ds4_session_snapshot_free(&initial);ds4_session_snapshot_free(&final);
    ds4_session_free(s);ds4_tokens_free(&prompt);ds4_engine_close(e);return 0;
}
