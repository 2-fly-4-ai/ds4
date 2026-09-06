#include "../ds4.c"
static void check(bool ok, const char *msg) {
    if (!ok) { fprintf(stderr, "MATRIX_FAIL %s\n", msg); exit(2); }
}
static char *read_text(const char *path) {
    FILE *f=fopen(path,"rb"); check(f!=NULL,path);
    fseek(f,0,SEEK_END); long n=ftell(f); rewind(f);
    char *p=calloc((size_t)n+1,1); check(p&&fread(p,1,n,f)==(size_t)n,"read");
    fclose(f); return p;
}
static ds4_session *new_session(ds4_engine *e, int ctx, int mode) {
    check(mode == 0, "production history disabled");
    ds4_session *s = NULL;
    check(ds4_session_create(&s, e, ctx) == 0, "session");
    return s;
}
static void sync_tokens(ds4_session *s, const ds4_tokens *p) {
    char err[512]={0};
    if(getenv("MATRIX_SCALAR_PREFILL")&&!s->checkpoint_valid) {
        ds4_tokens first=*p;first.len=1;
        check(ds4_session_sync(s,&first,err,sizeof(err))==0,err);
        for(int i=1;i<p->len;i++)check(ds4_session_eval(s,p->v[i],err,sizeof(err))==0,err);
    } else check(ds4_session_sync(s,p,err,sizeof(err))==0,err);
}
static void save_snapshot(ds4_session *s, ds4_session_snapshot *snap) {
    char err[512]={0};check(ds4_session_save_snapshot(s,snap,err,sizeof(err))==0,err);
}
static void restore_snapshot(ds4_session *s, const ds4_session_snapshot *snap) {
    char err[512]={0};check(ds4_session_load_snapshot(s,snap,err,sizeof(err))==0,err);
}
static bool exact_logits(ds4_session *a,ds4_session *b){
    return !memcmp(a->logits,b->logits,DS4_N_VOCAB*sizeof(float));
}
static bool same_tensor(ds4_gpu_tensor *a,ds4_gpu_tensor *b,size_t n){
    if(!n)return true;
    void *x=malloc(n),*y=malloc(n);check(x&&y,"state allocation");
    if(!ds4_gpu_tensor_read(a,0,x,n)||!ds4_gpu_tensor_read(b,0,y,n)){
        printf("STATE_READ_ERROR wanted=%zu a_bytes=%llu b_bytes=%llu\n",n,(unsigned long long)ds4_gpu_tensor_bytes(a),(unsigned long long)ds4_gpu_tensor_bytes(b));free(x);free(y);return false;
    }
    bool same=!memcmp(x,y,n);free(x);free(y);return same;
}
static bool same_state(ds4_session *a,ds4_session *b,uint32_t visible){
    check(ds4_gpu_synchronize(),"state synchronize");
    ds4_glm_gpu_graph *g=&a->glm_graph,*h=&b->glm_graph;
    printf("STATE_RANGE first=%u last=%u model_layers=%u\n",g->layer_start,g->layer_end,DS4_N_LAYER);
    for(uint32_t il=0;il<DS4_N_LAYER;il++){
        if(il<g->layer_start||il>g->layer_end)continue;
        bool same;
        if(ds4_glm53_layer_is_kda(il)){
            same=same_tensor(g->layer_kda_conv_state[il],h->layer_kda_conv_state[il],session_glm_kda_conv_state_bytes())&&
                 same_tensor(g->layer_kda_recurrent_state[il],h->layer_kda_recurrent_state[il],session_glm_kda_recurrent_state_bytes());
        }else{
            size_t element=glm_graph_compact_cache_is_f16()?2:4;
            same=same_tensor(g->layer_kv_lora_cache[il],h->layer_kv_lora_cache[il],(size_t)visible*DS4_N_KV_LORA*element)&&
                 same_tensor(g->layer_k_rope_cache[il],h->layer_k_rope_cache[il],(size_t)visible*DS4_N_ROT*element);
        }
        if(!same){printf("STATE_DIFF layer=%u\n",il);return false;}
    }
    return true;
}
static bool same_pool(ds4_session *a,ds4_session *b,uint32_t visible){
    check(ds4_gpu_synchronize(),"pool sync");
    ds4_glm_gpu_graph *g=&a->glm_graph,*h=&b->glm_graph;
    for(uint32_t il=g->layer_start;il<=g->layer_end;il++){
        if(!glm_graph_layer_uses_full_indexer(il))continue;
        size_t rows=session_glm_index_live_rows(il,visible),width=DS4_N_INDEXER_HEAD_DIM;
        bool cache=same_tensor(g->layer_indexer_key_cache[il],h->layer_indexer_key_cache[il],rows*width*(glm_graph_compact_cache_is_f16()?2:4));
        size_t tail=(visible%DS4_GLM53_INDEX_POOL_SIZE)*width*sizeof(float);
        bool key=same_tensor(g->layer_indexer_tail_k[il],h->layer_indexer_tail_k[il],tail);
        bool gate=same_tensor(g->layer_indexer_tail_gate[il],h->layer_indexer_tail_gate[il],tail);
        if(!cache||!key||!gate){printf("POOL_DIFF visible=%u layer=%u cache=%d key=%d gate=%d\n",visible,il,cache,key,gate);return false;}
    }
    return true;
}

/* Greedy: token identity plus full logits/state after every public route.
 * Sampled: teacher-force the returned tokens into scalar, checking the same
 * logits/state contract. Native MTP's accept/reject sampler consumes different
 * RNG draws from scalar CDF sampling; equal seeds need not yield equal text.
 * This is not a statistical validation of the sampling distribution. */
int main(int argc,char **argv){
    if(argc<3)return 2;
    const int ctx=getenv("ROUTER_CTX")?atoi(getenv("ROUTER_CTX")):8192;
    const int length=getenv("ROUTER_LENGTH")?atoi(getenv("ROUTER_LENGTH")):0;
    const float temperature=getenv("ROUTER_TEMP")?atof(getenv("ROUTER_TEMP")):0;
    ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,
        .context_size=ctx,.power_percent=100,.warm_weights=true,.glm_mtp=true,.dspark_exact_sampling=true};
    ds4_engine *e=NULL;check(ds4_engine_open(&e,&opt)==0,"engine");char err[512]={0};int bad=0;
    const int modes=1;
    for(int f=2;f<argc;f++)for(int mode=0;mode<modes;mode++){
        char *text=read_text(argv[f]);ds4_tokens p={0};ds4_encode_chat_prompt(e,NULL,text,DS4_THINK_NONE,&p);free(text);
        if(length>p.len){
            ds4_tokens padded={0},pad={0};ds4_tokenize_text(e,"Background note: preserve existing behavior.\n",&pad);
            for(int i=0;i<length-p.len;i++)token_vec_push(&padded,pad.v[i%pad.len]);
            for(int i=0;i<p.len;i++)token_vec_push(&padded,p.v[i]);
            ds4_tokens_free(&p);ds4_tokens_free(&pad);p=padded;
        }
        check(ctx>=p.len+256,"context capacity");
        ds4_session *s=new_session(e,ctx,mode),*r=new_session(e,ctx,0);sync_tokens(s,&p);sync_tokens(r,&p);
        int total=0,calls=0,restores=0,seed_differences=0;bool pass=exact_logits(s,r);uint64_t rng=1234,ref_rng=1234;
        if(!pass)bad++;
        while(pass&&total<256){
            if(getenv("ROUTER_RESTORE")&&total>=128&&!restores){
                ds4_session_snapshot snap={0};save_snapshot(s,&snap);
                for(int i=0;i<3;i++)check(ds4_session_eval(s,ds4_session_argmax(s),err,sizeof(err))==0,err);
                restore_snapshot(s,&snap);ds4_session_snapshot_free(&snap);restores++;
                pass=exact_logits(s,r)&&same_state(s,r,r->checkpoint.len)&&same_pool(s,r,r->checkpoint.len);
                if(!pass){bad++;break;}
            }
            int first=ds4_session_sample(s,temperature,40,.95,.05,&rng);if(ds4_token_is_stop(e,first))break;
            int out[16],left=256-total,cap=left<16?left:16,n;
            int route=ds4_session_select_decode_route(s,first,true,true,false);
            ds4_tokens before={0};ds4_tokens_copy(&before,&r->checkpoint);
            if(route==DS4_DECODE_ROUTE_PROMPT_LOOKUP)n=ds4_session_eval_prompt_lookup_sampled(s,first,left,ds4_token_eos(e),temperature,40,.95,.05,&rng,out,cap,err,sizeof(err));
            else if(route==DS4_DECODE_ROUTE_NEURAL_SPECULATION)n=ds4_session_eval_speculative(s,first,left,ds4_token_eos(e),temperature,40,.95,.05,&rng,out,cap,err,sizeof(err));
            else{out[0]=first;n=1;check(ds4_session_eval(s,first,err,sizeof(err))==0,err);}
            check(n>0&&n<=cap,err);
            for(int i=0;i<n;i++){
                int expected=ds4_session_sample(r,temperature,40,.95,.05,&ref_rng);
                if(out[i]!=expected)seed_differences++;
                if(temperature<=0)pass=pass&&out[i]==expected;
                check(ds4_session_eval(r,out[i],err,sizeof(err))==0,err);
            }
            pass=pass&&exact_logits(s,r);
            if(getenv("ROUTER_POOL_EACH"))pass=same_pool(s,r,r->checkpoint.len)&&pass;
            printf("BLOCK file=%d mode=%d offset=%d route=%d returned=%d pass=%d\n",f-2,mode,total,route,n,pass);fflush(stdout);
            total+=n;calls++;
            if(!pass){
                printf("FAIL_DETAIL logits_exact=%d rng_exact=%d\n",exact_logits(s,r),rng==ref_rng);
                same_state(s,r,r->checkpoint.len);same_pool(s,r,r->checkpoint.len);
                char name[256];snprintf(name,sizeof(name),"/tmp/glm-router-failure-l%d-t%.1f-%d-%d.tokens",p.len,temperature,f-2,mode);
                FILE *fp=fopen(name,"wb");check(fp!=NULL,"fixture");check(fwrite(before.v,4,before.len,fp)==(size_t)before.len,"fixture write");fclose(fp);bad++;
            }
            ds4_tokens_free(&before);
        }
        if(pass&&(!same_state(s,r,r->checkpoint.len)||!same_pool(s,r,r->checkpoint.len))){pass=false;bad++;}
        printf("ROUTER_PARITY file=%d mode=%d length=%d temp=%.2f tokens=%d calls=%d restores=%d seed_differences=%d pass=%d\n",f-2,mode,p.len,temperature,total,calls,restores,seed_differences,pass);fflush(stdout);
        ds4_session_free(s);ds4_session_free(r);ds4_tokens_free(&p);
    }
    printf("ROUTER_PARITY_COMPLETE failures=%d\n",bad);ds4_engine_close(e);return bad?1:0;
}
