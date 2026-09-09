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

typedef struct {
    uint64_t logits_end;
    uint64_t target_begin;
    uint64_t mtp_begin;
    uint64_t mtp_end;
    uint64_t suffix_begin;
} qwen_snapshot_layout;

static bool qwen_snapshot_layout_read(
        const ds4_session *s, const ds4_session_snapshot *snap,
        qwen_snapshot_layout *layout) {
    if (!s || !snap || !snap->ptr || !layout || !ds4_session_is_qwen4(s)) {
        return false;
    }
    const uint32_t rows = (uint32_t)s->checkpoint.len;
    const uint64_t logits_end =
        (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) +
        (uint64_t)rows * sizeof(uint32_t) +
        (uint64_t)DS4_N_VOCAB * sizeof(float);
    if (logits_end + sizeof(uint32_t) > snap->len) return false;
    uint32_t mtp_rows = 0;
    memcpy(&mtp_rows, snap->ptr + logits_end, sizeof(mtp_rows));
    if (mtp_rows > rows) return false;
    uint64_t off = logits_end + sizeof(uint32_t);
    uint64_t mtp_begin = UINT64_MAX, mtp_end = UINT64_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        uint64_t bytes = 0;
        if (ds4_qwen4_layer_is_linear(il)) {
            bytes = qwen4_payload_lin_state_bytes() +
                    qwen4_payload_lin_hist_bytes();
        } else {
            const uint32_t n = ds4_qwen4_layer_is_nextn(il) ? mtp_rows : rows;
            bytes = 2u * qwen4_payload_kv_bytes(n) +
                    qwen4_payload_ik_bytes(n) +
                    qwen4_payload_block_key_bytes(n);
        }
        if (ds4_qwen4_layer_is_nextn(il)) mtp_begin = off;
        off += bytes;
        if (ds4_qwen4_layer_is_nextn(il)) mtp_end = off;
    }
    if (mtp_begin == UINT64_MAX || mtp_end == UINT64_MAX || off > snap->len) {
        return false;
    }
    *layout = (qwen_snapshot_layout){
        .logits_end = logits_end,
        .target_begin = logits_end + sizeof(uint32_t),
        .mtp_begin = mtp_begin,
        .mtp_end = mtp_end,
        .suffix_begin = off,
    };
    return true;
}

/* A verifier may produce a different disposable proposal cache, but emitted
 * tokens, target logits, trunk state, PLE history and positions stay exact.
 * Restoring a payload clears glm_mtp_have, so no proposal survives reload. */
static void same_qwen_target_snapshot(
        ds4_session *s, const ds4_session_snapshot *expected) {
    char err[512] = {0};
    ds4_session_snapshot actual = {0};
    ck(ds4_session_save_snapshot(s, &actual, err, sizeof(err)) == 0, err);
    qwen_snapshot_layout a = {0}, e = {0};
    ck(qwen_snapshot_layout_read(s, &actual, &a) &&
       qwen_snapshot_layout_read(s, expected, &e),
       "Qwen target snapshot layout");
    ck(a.logits_end == e.logits_end &&
       a.mtp_begin - a.target_begin == e.mtp_begin - e.target_begin &&
       actual.len - a.mtp_end == expected->len - e.mtp_end,
       "Qwen target snapshot spans");
    if(memcmp(actual.ptr,expected->ptr,a.logits_end)) {
        uint64_t first=0;
        while(first<a.logits_end&&actual.ptr[first]==expected->ptr[first])first++;
        const uint64_t logits_begin=a.logits_end-(uint64_t)DS4_N_VOCAB*sizeof(float);
        float worst=0.0f;unsigned changed=0;
        const float *got_logits=(const float *)(actual.ptr+logits_begin);
        const float *want_logits=(const float *)(expected->ptr+logits_begin);
        for(uint32_t v=0;v<DS4_N_VOCAB;v++){
            const float d=fabsf(got_logits[v]-want_logits[v]);
            if(d>worst)worst=d;
            changed+=d!=0.0f;
        }
        fprintf(stderr,"QWEN_TARGET_PREFIX_DIFF first=%llu logits_begin=%llu\n",
                (unsigned long long)first,
                (unsigned long long)logits_begin);
        fprintf(stderr,"QWEN_TARGET_LOGITS changed=%u max=%g want=%d got=%d\n",
                changed,worst,sample_argmax(want_logits,DS4_N_VOCAB),
                sample_argmax(got_logits,DS4_N_VOCAB));
        ck(false,"Qwen target tokens/logits");
    }
    if(memcmp(actual.ptr + a.target_begin,expected->ptr + e.target_begin,
              a.mtp_begin - a.target_begin)) {
        uint64_t first=0,n=a.mtp_begin-a.target_begin;
        while(first<n&&actual.ptr[a.target_begin+first]==expected->ptr[e.target_begin+first])first++;
        fprintf(stderr,"QWEN_TARGET_TRUNK_DIFF first=%llu\n",
                (unsigned long long)first);
        ck(false,"Qwen target trunk state");
    }
    if(memcmp(actual.ptr + a.mtp_end,expected->ptr + e.mtp_end,
              actual.len - a.mtp_end)) {
        uint64_t first=0,n=actual.len-a.mtp_end;
        while(first<n&&actual.ptr[a.mtp_end+first]==expected->ptr[e.mtp_end+first])first++;
        fprintf(stderr,"QWEN_TARGET_SUFFIX_DIFF first=%llu\n",
                (unsigned long long)first);
        ck(false,"Qwen target suffix state");
    }
    if (a.mtp_end - a.mtp_begin != e.mtp_end - e.mtp_begin ||
        memcmp(actual.ptr + a.mtp_begin, expected->ptr + e.mtp_begin,
               a.mtp_end - a.mtp_begin)) {
        fprintf(stderr,
                "QWEN_PROPOSAL_STATE_DIFF expected_bytes=%llu actual_bytes=%llu (allowed diagnostic)\n",
                (unsigned long long)(e.mtp_end - e.mtp_begin),
                (unsigned long long)(a.mtp_end - a.mtp_begin));
    }
    ds4_session_snapshot_free(&actual);
}

static int batch_scalar_probe(const char *model_path) {
    const char *task = getenv("HOT_TASK");
    const char *suffix_text = getenv("HOT_BATCH_SCALAR_SUFFIX");
    const char *logit_path = getenv("HOT_BATCH_SCALAR_LOGITS");
    const bool batch = getenv("HOT_BATCH_SCALAR_BATCH") != NULL;
    int width = getenv("HOT_BATCH_SCALAR_WIDTH") ?
        atoi(getenv("HOT_BATCH_SCALAR_WIDTH")) : 4;
    ck(task && task[0], "batch/scalar task");
    ck(suffix_text && suffix_text[0], "batch/scalar suffix");
    ck(width >= 2 && width <= 16, "batch/scalar width 2..16");

    ds4_engine_options opt = {
        .model_path = model_path,
        .backend = DS4_BACKEND_METAL,
        .context_size = 4096,
        .power_percent = 100,
        .warm_weights = getenv("HOT_NO_WARM") == NULL,
        .glm_mtp = false,
    };
    ds4_engine *e = NULL;
    ck(ds4_engine_open(&e, &opt) == 0, "batch/scalar engine");
    ds4_tokens prompt = {0}, suffix = {0};
    ds4_encode_chat_prompt(e, NULL, task, DS4_THINK_NONE, &prompt);
    ds4_tokenize_text(e, suffix_text, &suffix);
    ck(prompt.len > 0, "batch/scalar prompt");
    ck(suffix.len >= width, "batch/scalar suffix token count");

    ds4_session *s = NULL;
    char err[512] = {0};
    ck(ds4_session_create(&s, e, opt.context_size) == 0,
       "batch/scalar session");
    ck(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0, err);
    ck(ds4_session_is_qwen4(s), "batch/scalar Qwen model");
    const ds4_layer_weights *l0 = &e->weights.layer[0];
    printf("QWEN_BATCH_SCALAR_TYPES hc_down=%u hc_up=%u hc_inject=%u "
           "lin_qkv=%u expert_gate=%u expert_down=%u\n",
           l0->hc_attn_down->type, l0->hc_attn_up->type,
           l0->hc_attn_inject->type, l0->lin_qkv->type,
           l0->ffn_gate_exps->type, l0->ffn_down_exps->type);

    const uint32_t V = DS4_N_VOCAB;
    float *logits = malloc((size_t)width * V * sizeof(float));
    ck(logits != NULL, "batch/scalar logits");
    if (batch) {
        const bool exact = getenv("HOT_BATCH_SCALAR_EXACT") != NULL;
        ck((exact ? qwen4_graph_forward_verify_rows(
                        &s->qwen4_graph, &e->model, &e->weights,
                        suffix.v, (uint32_t)width, logits, true)
                  : qwen4_graph_forward_tokens(
                        &s->qwen4_graph, &e->model, &e->weights,
                        suffix.v, (uint32_t)width, logits, true)),
           "batch target forward");
    } else {
        for (int i = 0; i < width; i++) {
            ck(qwen4_graph_forward_tokens(
                   &s->qwen4_graph, &e->model, &e->weights, suffix.v + i, 1,
                   logits + (size_t)i * V, false),
               "scalar target forward");
        }
    }
    if (logit_path && logit_path[0]) {
        ck(write_f32_binary_file(logit_path, logits, (uint64_t)width * V),
           "batch/scalar logit write");
    }
    printf("QWEN_BATCH_SCALAR mode=%s prompt=%d width=%d pos=%u",
           batch ? "batch" : "scalar", prompt.len, width,
           s->qwen4_graph.pos);
    for (int i = 0; i < width; i++) {
        printf(" row%d=%d", i,
               sample_argmax(logits + (size_t)i * V, V));
    }
    putchar('\n');
    fflush(stdout);

    free(logits);
    ds4_session_free(s);
    ds4_tokens_free(&suffix);
    ds4_tokens_free(&prompt);
    ds4_engine_close(e);
    return 0;
}

int main(int argc,char **argv) {
    ck(argc==2&&getenv("HOT_TASK"),"model/task");
    if(getenv("HOT_BATCH_SCALAR"))return batch_scalar_probe(argv[1]);
    int ctx=getenv("HOT_CTX")?atoi(getenv("HOT_CTX")):2048;
    int candidate=getenv("HOT_VARIANT")?atoi(getenv("HOT_VARIANT")):8;
    int budget=getenv("HOT_TOKENS")?atoi(getenv("HOT_TOKENS")):128;
    float temp=getenv("HOT_TEMP")?atof(getenv("HOT_TEMP")):0;
    bool lookup=getenv("HOT_LOOKUP")!=NULL;
    bool neural_allowed=getenv("HOT_PLAIN")==NULL;
    bool relaxed_route=getenv("HOT_RELAX_ROUTE")!=NULL;
    bool reference_plain=getenv("HOT_REFERENCE_PLAIN")!=NULL;
    ck(budget>0&&budget<=512,"budget");variant(reference_plain?candidate:0);
    ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,.context_size=ctx+1024,
        .power_percent=100,.warm_weights=getenv("HOT_NO_WARM")==NULL,.glm_mtp=neural_allowed,.dspark_exact_sampling=true};
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
    ds4_session_snapshot initial={0},final={0},candidate_final={0},rewound[2]={{0},{0}};ck(ds4_session_save_snapshot(s,&initial,err,sizeof(err))==0,err);
    int tokens[512],counts[512]={0},routes[512]={0};
    float *logits=malloc((size_t)budget*DS4_N_VOCAB*4),*drafts=malloc((size_t)budget*DS4_N_VOCAB*4);
    ck(logits&&drafts,"oracles");int draft_ids[512],parents[512],have[512];
    for(int arm=0;arm<2;arm++) {
        /* Exercise both arms from the serialized checkpoint.  Deep MTP can be
         * more sensitive than the one-token path to a live-prefill versus
         * restored-cache mismatch, so do not use unlike starting states. */
        ck(ds4_session_load_snapshot(s,&initial,err,sizeof(err))==0,err);
        variant(reference_plain?candidate:(arm?candidate:0));uint64_t rng=123;
        s->qwen4_spec_cycles=s->qwen4_spec_accepted=0;
        int neural=0,lookups=0;double arm_t0=now_sec();
        for(int n=0;n<budget;) {
            int first=ds4_session_sample(s,temp,40,.8f,0,&rng),accepted[32],cap=budget-n<32?budget-n:32;
            ds4_decode_route route=(!arm&&reference_plain)?DS4_DECODE_ROUTE_PLAIN:
                lookup?ds4_session_select_decode_route(s,first,true,true,false):
                neural_allowed?DS4_DECODE_ROUTE_NEURAL_SPECULATION:DS4_DECODE_ROUTE_PLAIN;
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
            }else if(relaxed_route){
                ck(!memcmp(tokens+n,accepted,count*sizeof(int)),"target token stream");
                if(getenv("HOT_RELAX_LOGITS")) {
                    ck(reference_plain,"relaxed logit oracle needs plain reference");
                    const float *want=logits+(size_t)(n+count-1)*DS4_N_VOCAB;
                    if(memcmp(want,s->logits,(size_t)DS4_N_VOCAB*sizeof(float))) {
                        float worst=0.0f;unsigned diff=0;
                        for(uint32_t v=0;v<DS4_N_VOCAB;v++) {
                            const float d=fabsf(want[v]-s->logits[v]);
                            if(d>worst)worst=d;
                            diff+=d!=0.0f;
                        }
                        fprintf(stderr,
                                "QWEN_TARGET_LOGIT_DIFF pos=%d count=%d changed=%u max=%g want_argmax=%d got_argmax=%d\n",
                                n+count,count,diff,worst,
                                sample_argmax(want,DS4_N_VOCAB),
                                sample_argmax(s->logits,DS4_N_VOCAB));
                        ck(false,"position-aligned target logits");
                    }
                }
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
        double arm_elapsed=now_sec()-arm_t0;
        printf("QWEN_PORT_ARM variant=%d arm=%d prompt=%d temp=%g neural=%d lookup=%d cycles=%llu accepted=%llu ms=%.3f tps=%.3f\n",candidate,arm,prompt.len,temp,neural,lookups,(unsigned long long)s->qwen4_spec_cycles,(unsigned long long)s->qwen4_spec_accepted,arm_elapsed*1000.0,budget/arm_elapsed);fflush(stdout);
        if(!arm)ck(ds4_session_save_snapshot(s,&final,err,sizeof(err))==0,err);
        else if(relaxed_route) {
            if(getenv("HOT_DIAGNOSTIC_ALLOW_TARGET_DRIFT")) {
                fprintf(stderr,"QWEN_TARGET_DRIFT_ALLOWED diagnostic timing only\n");
            } else {
                same_qwen_target_snapshot(s,&final);
            }
            ck(ds4_session_save_snapshot(s,&candidate_final,err,sizeof(err))==0,err);
        }
        else same_snapshot(s,&final);
        if(getenv("HOT_REWIND")) {
            ck(budget>=2,"rewind budget");
            for(int back=1;back<=2;back++) {
                const uint32_t rewind_pos=(uint32_t)(prompt.len+budget-back);
                const bool fast_rewind=arm&&relaxed_route&&
                    s->qwen4_graph.snap_valid&&
                    s->qwen4_graph.snap_pos==rewind_pos&&
                    s->qwen4_rewind_valid&&s->qwen4_rewind_pos==rewind_pos;
                ds4_session_rewind(s,prompt.len+budget-back);
                ck(ds4_session_pos(s)==prompt.len+budget-back,"rewind position");
                for(int j=budget-back;j<budget;j++)ck(ds4_session_eval(s,tokens[j],err,sizeof(err))==0,err);
                if(fast_rewind) {
                    fprintf(stderr,"QWEN_PORT_REWIND_ORIGINAL_COMPARE back=%d\n",back);
                    same_qwen_target_snapshot(s,&candidate_final);
                }
                if(!arm)ck(ds4_session_save_snapshot(s,&rewound[back-1],err,sizeof(err))==0,err);
                else if(!relaxed_route) same_snapshot(s,&rewound[back-1]);
            }
            printf("QWEN_PORT_REWIND_EXACT variant=%d arm=%d\n",candidate,arm);fflush(stdout);
        }
    }
    if(relaxed_route&&!getenv("HOT_DIAGNOSTIC_ALLOW_TARGET_DRIFT")) {
        /* A payload restore discards any pending proposal.  Continue both
         * target-equivalent snapshots through the same release kernels and
         * prove that their disposable MTP cache cannot affect future output. */
        int continuation[8];
        float *continuation_logits=malloc(8u*(size_t)DS4_N_VOCAB*sizeof(float));
        ck(continuation_logits,"continuation logits");
        for(int arm=0;arm<2;arm++) {
            const ds4_session_snapshot *src=arm?&candidate_final:&final;
            ck(ds4_session_load_snapshot(s,src,err,sizeof(err))==0,err);
            variant(candidate);
            for(int i=0;i<8;i++) {
                const int tok=ds4_session_argmax(s);
                if(!arm) {
                    continuation[i]=tok;
                    memcpy(continuation_logits+(size_t)i*DS4_N_VOCAB,
                           s->logits,(size_t)DS4_N_VOCAB*sizeof(float));
                } else {
                    ck(tok==continuation[i],"restored continuation token");
                    ck(!memcmp(continuation_logits+(size_t)i*DS4_N_VOCAB,
                               s->logits,(size_t)DS4_N_VOCAB*sizeof(float)),
                       "restored continuation logits");
                }
                ck(ds4_session_eval(s,tok,err,sizeof(err))==0,err);
            }
        }
        free(continuation_logits);
        printf("QWEN_PORT_RESTORE_CONTINUATION_PASS variant=%d tokens=8\n",candidate);
        fflush(stdout);
    }
    printf("QWEN_PORT_PARITY_PASS variant=%d tokens=%d prompt=%d snapshot=%llu\n",candidate,budget,prompt.len,(unsigned long long)final.len);
    if(getenv("HOT_TIMING")) {
        const int order[]={0,1,1,0,1,0,0,1};
        for(int repeat=getenv("HOT_STEADY")?-8:0;repeat<8;repeat++) {
            const int arm=order[(repeat+8)%8];
            ck(ds4_session_load_snapshot(s,&initial,err,sizeof(err))==0,err);
            variant(reference_plain?candidate:(arm?candidate:0));uint64_t rng=123;
            s->qwen4_spec_cycles=s->qwen4_spec_accepted=0;
            int generated[512],n=0;double start=now_sec();
            while(n<budget) {
                int first=ds4_session_sample(s,temp,40,.8f,0,&rng),accepted[32],cap=budget-n<32?budget-n:32;
                ds4_decode_route route=(!arm&&reference_plain)?DS4_DECODE_ROUTE_PLAIN:
                    lookup?ds4_session_select_decode_route(s,first,true,true,false):
                    neural_allowed?DS4_DECODE_ROUTE_NEURAL_SPECULATION:DS4_DECODE_ROUTE_PLAIN;
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
    free(logits);free(drafts);ds4_session_snapshot_free(&initial);ds4_session_snapshot_free(&final);ds4_session_snapshot_free(&candidate_final);
    ds4_session_snapshot_free(&rewound[0]);ds4_session_snapshot_free(&rewound[1]);
    ds4_session_free(s);ds4_tokens_free(&prompt);ds4_engine_close(e);return 0;
}
