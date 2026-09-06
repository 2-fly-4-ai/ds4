#define main original_hot_main
#include "test_ds4_hot_rewind.c"
#undef main
static void variant(int mode) {
    /* Preserve experiment IDs in the archived logs: 0=reference, 2=HC,
     * 3=GDN, 6=HC+GDN, 7=serial predictor, 8=retained decode bundle.
     * Rejected IDs 1/4/5 are deliberately no longer executable. */
    ck(mode==0||mode==2||mode==3||mode==6||mode==7||mode==8,"retained variant required");
    setenv("DS4_QWEN_MTP_SERIAL_DISABLE","1",1);
    setenv("DS4_QWEN_HC_PAIR_DISABLE","1",1);
    setenv("DS4_QWEN_GDN_R4_DISABLE","1",1);
    setenv("DS4_QWEN_NORM_REUSE_DISABLE","1",1);
    if(mode==7||mode==8)unsetenv("DS4_QWEN_MTP_SERIAL_DISABLE");
    if(mode==2||mode==6||mode==8)unsetenv("DS4_QWEN_HC_PAIR_DISABLE");
    if(mode==3||mode==6||mode==8)unsetenv("DS4_QWEN_GDN_R4_DISABLE");
}
int main(int argc,char **argv) {
    ck(argc==2&&getenv("HOT_TASK"),"model/task");
    int ctx=getenv("HOT_CTX")?atoi(getenv("HOT_CTX")):2048;
    int candidate=getenv("HOT_VARIANT")?atoi(getenv("HOT_VARIANT")):8;
    int budget=getenv("HOT_TOKENS")?atoi(getenv("HOT_TOKENS")):128;
    float temp=getenv("HOT_TEMP")?atof(getenv("HOT_TEMP")):0;
    bool lookup=getenv("HOT_LOOKUP")!=NULL;
    bool neural_allowed=getenv("HOT_PLAIN")==NULL;
    ck(budget>0&&budget<=512,"budget");variant(0);
    ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,.context_size=ctx+1024,
        .power_percent=100,.warm_weights=true,.glm_mtp=neural_allowed,.dspark_exact_sampling=true};
    ds4_engine *e=NULL;ck(ds4_engine_open(&e,&opt)==0,"engine");
    ds4_tokens prompt={0};ds4_encode_chat_prompt(e,NULL,getenv("HOT_TASK"),DS4_THINK_NONE,&prompt);
    if(getenv("HOT_PAD")) {
        ds4_tokens pad={0};ds4_tokenize_text(e," Archive entry: keep verified records and preserve revision history.",&pad);
        ds4_tokens p={0};for(int i=0;i<ctx-prompt.len;i++)token_vec_push(&p,pad.v[i%pad.len]);
        for(int i=0;i<prompt.len;i++)token_vec_push(&p,prompt.v[i]);
        ds4_tokens_free(&prompt);ds4_tokens_free(&pad);prompt=p;
    }
    ds4_session *s=NULL;char err[512]={0};ck(ds4_session_create(&s,e,opt.context_size)==0,"session");
    ck(ds4_session_sync(s,&prompt,err,sizeof(err))==0,err);
    ds4_session_snapshot initial={0},final={0},rewound[2]={{0},{0}};ck(ds4_session_save_snapshot(s,&initial,err,sizeof(err))==0,err);
    int tokens[512],counts[512]={0},routes[512]={0};
    float *logits=malloc((size_t)budget*DS4_N_VOCAB*4),*drafts=malloc((size_t)budget*DS4_N_VOCAB*4);
    ck(logits&&drafts,"oracles");int draft_ids[512],parents[512],have[512];
    for(int arm=0;arm<2;arm++) {
        if(arm)ck(ds4_session_load_snapshot(s,&initial,err,sizeof(err))==0,err);
        variant(arm?candidate:0);uint64_t rng=123;
        s->qwen4_spec_cycles=s->qwen4_spec_accepted=0;
        int neural=0,lookups=0;
        for(int n=0;n<budget;) {
            int first=ds4_session_sample(s,temp,40,.8f,0,&rng),accepted[32],cap=budget-n<32?budget-n:32;
            ds4_decode_route route=lookup?ds4_session_select_decode_route(s,first,true,true,false):neural_allowed?DS4_DECODE_ROUTE_NEURAL_SPECULATION:DS4_DECODE_ROUTE_PLAIN;
            int count;
            if(route==DS4_DECODE_ROUTE_PROMPT_LOOKUP){
                count=ds4_session_eval_prompt_lookup_sampled(s,first,budget-n,ds4_token_eos(e),temp,40,.8f,0,&rng,accepted,cap,err,sizeof(err));lookups++;
            }else if(route==DS4_DECODE_ROUTE_NEURAL_SPECULATION){
                count=ds4_session_eval_speculative(s,first,budget-n,ds4_token_eos(e),temp,40,.8f,0,&rng,accepted,cap,err,sizeof(err));neural++;
            }else{ck(ds4_session_eval(s,first,err,sizeof(err))==0,err);accepted[0]=first;count=1;}
            ck(count>0&&count<=cap,err);
            if(!arm){
                counts[n]=count;routes[n]=route;memcpy(tokens+n,accepted,count*sizeof(int));
                memcpy(logits+(size_t)n*DS4_N_VOCAB,s->logits,DS4_N_VOCAB*4);
                if(s->glm_mtp_have)memcpy(drafts+(size_t)n*DS4_N_VOCAB,s->qwen4_graph.host_logits,DS4_N_VOCAB*4);
                draft_ids[n]=s->glm_mtp_draft;parents[n]=s->glm_mtp_parent;have[n]=s->glm_mtp_have;
            }else{
                ck(counts[n]==count&&routes[n]==(int)route&&!memcmp(tokens+n,accepted,count*sizeof(int)),"route/token counts");
                ck(!memcmp(logits+(size_t)n*DS4_N_VOCAB,s->logits,DS4_N_VOCAB*4),"target full logits");
                ck(have[n]==s->glm_mtp_have,"draft validity");
                if(s->glm_mtp_have){
                    ck(draft_ids[n]==s->glm_mtp_draft&&parents[n]==s->glm_mtp_parent,"draft/parent ids");
                    if(memcmp(drafts+(size_t)n*DS4_N_VOCAB,s->qwen4_graph.host_logits,DS4_N_VOCAB*4)) {
                        float worst=0;unsigned diff=0;for(uint32_t v=0;v<DS4_N_VOCAB;v++){float d=fabsf(drafts[(size_t)n*DS4_N_VOCAB+v]-s->qwen4_graph.host_logits[v]);if(d>worst)worst=d;diff+=d!=0;}
                        fprintf(stderr,"QWEN_PORT_DRAFT_DIFF variant=%d step=%d count=%u max=%g\n",candidate,n,diff,worst);ck(false,"draft full logits");
                    }
                }
            }
            n+=count;
        }
        printf("QWEN_PORT_ARM variant=%d arm=%d prompt=%d temp=%g neural=%d lookup=%d cycles=%llu accepted=%llu\n",candidate,arm,prompt.len,temp,neural,lookups,(unsigned long long)s->qwen4_spec_cycles,(unsigned long long)s->qwen4_spec_accepted);fflush(stdout);
        if(!arm)ck(ds4_session_save_snapshot(s,&final,err,sizeof(err))==0,err);else same_snapshot(s,&final);
        if(getenv("HOT_REWIND")) {
            ck(budget>=2,"rewind budget");
            for(int back=1;back<=2;back++) {
                ds4_session_rewind(s,prompt.len+budget-back);
                ck(ds4_session_pos(s)==prompt.len+budget-back,"rewind position");
                for(int j=budget-back;j<budget;j++)ck(ds4_session_eval(s,tokens[j],err,sizeof(err))==0,err);
                if(!arm)ck(ds4_session_save_snapshot(s,&rewound[back-1],err,sizeof(err))==0,err);
                else same_snapshot(s,&rewound[back-1]);
            }
            printf("QWEN_PORT_REWIND_EXACT variant=%d arm=%d\n",candidate,arm);fflush(stdout);
        }
    }
    printf("QWEN_PORT_PARITY_PASS variant=%d tokens=%d prompt=%d snapshot=%llu\n",candidate,budget,prompt.len,(unsigned long long)final.len);
    if(getenv("HOT_TIMING")) {
        const int order[]={0,1,1,0,1,0,0,1};
        for(int repeat=getenv("HOT_STEADY")?-8:0;repeat<8;repeat++) {
            const int arm=order[(repeat+8)%8];
            ck(ds4_session_load_snapshot(s,&initial,err,sizeof(err))==0,err);
            variant(arm?candidate:0);uint64_t rng=123;
            s->qwen4_spec_cycles=s->qwen4_spec_accepted=0;
            int generated[512],n=0;double start=now_sec();
            while(n<budget) {
                int first=ds4_session_sample(s,temp,40,.8f,0,&rng),accepted[32],cap=budget-n<32?budget-n:32;
                ds4_decode_route route=lookup?ds4_session_select_decode_route(s,first,true,true,false):neural_allowed?DS4_DECODE_ROUTE_NEURAL_SPECULATION:DS4_DECODE_ROUTE_PLAIN;
                int count;
                if(route==DS4_DECODE_ROUTE_PROMPT_LOOKUP)count=ds4_session_eval_prompt_lookup_sampled(s,first,budget-n,ds4_token_eos(e),temp,40,.8f,0,&rng,accepted,cap,err,sizeof(err));
                else if(route==DS4_DECODE_ROUTE_NEURAL_SPECULATION)count=ds4_session_eval_speculative(s,first,budget-n,ds4_token_eos(e),temp,40,.8f,0,&rng,accepted,cap,err,sizeof(err));
                else{ck(ds4_session_eval(s,first,err,sizeof(err))==0,err);accepted[0]=first;count=1;}
                ck(count>0&&count<=cap,err);memcpy(generated+n,accepted,count*sizeof(int));n+=count;
            }
            double elapsed=now_sec()-start;ck(!memcmp(tokens,generated,budget*sizeof(int)),"timed output parity");
            printf("QWEN_PORT_%s variant=%d mode=%d repeat=%d prompt=%d tokens=%d ms=%.4f tps=%.4f cycles=%llu accepted=%llu\n",repeat<0?"WARMUP":"TIMING",candidate,arm,repeat,prompt.len,budget,elapsed*1000,budget/elapsed,(unsigned long long)s->qwen4_spec_cycles,(unsigned long long)s->qwen4_spec_accepted);fflush(stdout);
        }
    }
    free(logits);free(drafts);ds4_session_snapshot_free(&initial);ds4_session_snapshot_free(&final);
    ds4_session_snapshot_free(&rewound[0]);ds4_session_snapshot_free(&rewound[1]);
    ds4_session_free(s);ds4_tokens_free(&prompt);ds4_engine_close(e);return 0;
}
