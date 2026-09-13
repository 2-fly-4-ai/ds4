/* Whole-model Metal correctness; bounded prompts, no CPU fallback. */
#include "../ds4.c"
static void check(int ok,const char *why){if(!ok){fprintf(stderr,"SMALL_RUNTIME_FAIL %s\n",why);exit(2);}}
static void run_model(const char *path) {
 ds4_engine_options opt={.model_path=path,.backend=DS4_BACKEND_METAL,
   .context_size=4096,.power_percent=100,.warm_weights=true};
 ds4_engine *e=NULL;check(!ds4_engine_open(&e,&opt),"engine");
 qwen_mtp_weights_t head={0};qwen_mtp_bind(&head,&e->model);
 check(qwen_mtp_is_valid(&head),"usable model-shaped MTP head");
 ds4_tensor wrong_norm=*head.enorm;wrong_norm.dim[0]++;
 head.enorm=&wrong_norm;check(!qwen_mtp_is_valid(&head),"reject wrong-width sidecar");
 ds4_tokens prompt={0},other={0};char err[512]={0};
 encode_chat_prompt(&e->vocab,NULL,"Write a Python function to merge two sorted lists. Explain its time complexity.",DS4_THINK_NONE,&prompt);
 encode_chat_prompt(&e->vocab,NULL,"Tell a story about a lost sailor.",DS4_THINK_NONE,&other);
 ds4_session *s=NULL,*b=NULL;check(!ds4_session_create(&s,e,4096),"session");
 check(!ds4_session_sync(s,&prompt,err,sizeof(err)),err);
 check(!qwen_hybrid_metal_forward_token_ex(NULL,NULL,NULL,NULL,NULL,0,&e->model,&e->weights,prompt.v[0],g_qwen_pool.max_ctx),"scalar cache bound");
 check(!qwen_hybrid_metal_forward_tokens(NULL,NULL,NULL,NULL,NULL,0,&e->model,&e->weights,prompt.v,2,g_qwen_pool.max_ctx-1,true,false),"verifier cache bound");
 check(ds4_session_payload_bytes(s)==0,"no invalid DS4-format persistence");
 FILE *disk=tmpfile();check(disk!=NULL,"temporary checkpoint");
 check(ds4_session_save_payload(s,disk,err,sizeof(err))!=0,"reject unsupported disk save");
 check(ds4_session_load_payload(s,disk,0,err,sizeof(err))!=0,"reject unsupported disk load");
 fclose(disk);
 float *initial=malloc(DS4_N_VOCAB*4ull);memcpy(initial,s->logits,DS4_N_VOCAB*4ull);
 check(ds4_session_mark_rewind_point(s),"mark");
 int tokens[96];
 for(int i=0;i<96;i++){tokens[i]=ds4_session_argmax(s);check(!ds4_session_eval(s,tokens[i],err,sizeof(err)),err);}
 check(!ds4_session_create(&b,e,4096),"other session");check(!ds4_session_sync(b,&other,err,sizeof(err)),err);
 check(g_qwen_pool.owner==b,"other owner");
 ds4_session_rewind(s,prompt.len);
 check(s->checkpoint_valid&&g_qwen_pool.owner==s,"restore ownership");
 check(!memcmp(initial,s->logits,DS4_N_VOCAB*4ull),"restore exact logits");
 for(int i=0;i<96;i++){check(tokens[i]==ds4_session_argmax(s),"repeated continuation");check(!ds4_session_eval(s,tokens[i],err,sizeof(err)),err);}
 ds4_session_rewind(s,prompt.len);
 for(int i=0;i<96;) {
   int accepted[8],n=ds4_session_eval_qwen_nextn_argmax(s,ds4_session_argmax(s),96-i,ds4_token_eos(e),accepted,8,err,sizeof(err));
   check(n>0,err);
   for(int j=0;j<n;j++)check(accepted[j]==tokens[i+j],"MTP versus serial tokens");
   i+=n;
 }
 ds4_session_rewind(s,prompt.len-1);check(!s->checkpoint_valid,"unmarked rewind invalidates");
 check(!ds4_session_sync(s,&other,err,sizeof(err)),err);
 check(!ds4_session_can_rewind_to(s,prompt.len),"new prompt invalidates old mark");
 check(s->mtp_probe_total>0&&s->mtp_probe_hit>0,"MTP actually drafted and accepted tokens");
 printf("SMALL_RUNTIME_PASS model=%s tokens=96 hot_bytes=%zu MTP_drafted=%llu accepted=%llu\n",
    path,s->qwen_hot_cap,(unsigned long long)s->mtp_probe_total,(unsigned long long)s->mtp_probe_hit);
 free(initial);ds4_session_free(b);ds4_session_free(s);ds4_tokens_free(&prompt);ds4_tokens_free(&other);ds4_engine_close(e);
 check(!g_qwen_pool.inited&&!g_mtp_pool.inited&&!g_qwen_mtp_sidecar_ready,"engine close clears global pools");
}
int main(int argc,char **argv) {
 check(argc>=2,"model paths");
 for(int i=1;i<argc;i++)run_model(argv[i]);
 return 0;
}
