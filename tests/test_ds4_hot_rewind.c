#include "../ds4.c"
static void ck(bool ok,const char *why){if(!ok){fprintf(stderr,"HOT_FAIL %s\n",why);exit(2);}}
static void same_snapshot(ds4_session *s,ds4_session_snapshot *expected){
 char err[512]={0};ds4_session_snapshot actual={0};ck(ds4_session_save_snapshot(s,&actual,err,sizeof(err))==0,err);
 ck(actual.len==expected->len&&!memcmp(actual.ptr,expected->ptr,actual.len),"payload state/logits differ");ds4_session_snapshot_free(&actual);
}
static void route_run(ds4_session *s,int *tokens,int *counts,int *routes,float *logits,bool compare){
 char err[512]={0};uint64_t rng=1234;
 int calls[3]={0},commits[3]={0};
 const float temp=getenv("HOT_TEMP")?atof(getenv("HOT_TEMP")):0;
 for(int n=0;n<128;){
  int first=ds4_session_sample(s,temp,40,.95f,.05f,&rng),accepted[32],cap=128-n<32?128-n:32;
  ds4_decode_route route=ds4_session_select_decode_route(s,first,true,true,false);
  int count;
  if(route==DS4_DECODE_ROUTE_PROMPT_LOOKUP)
   count=ds4_session_eval_prompt_lookup_sampled(s,first,128-n,ds4_token_eos(s->engine),temp,40,.95f,.05f,&rng,accepted,cap,err,sizeof(err));
  else if(route==DS4_DECODE_ROUTE_NEURAL_SPECULATION)
   count=ds4_session_eval_speculative(s,first,128-n,ds4_token_eos(s->engine),temp,40,.95f,.05f,&rng,accepted,cap,err,sizeof(err));
  else {ck(ds4_session_eval(s,first,err,sizeof(err))==0,err);accepted[0]=first;count=1;}
  ck(count>0&&count<=cap,err);
  calls[route]++;commits[route]+=count;
  if(compare){
   if(counts[n]!=count||routes[n]!=(int)route||memcmp(tokens+n,accepted,count*sizeof(int))||memcmp(logits+(size_t)n*DS4_N_VOCAB,s->logits,DS4_N_VOCAB*sizeof(float)))
    fprintf(stderr,"HOT_ROUTE_DIFF pos=%d count=%d expected=%d route=%d expected=%d\n",n,count,counts[n],route,routes[n]);
   ck(counts[n]==count&&routes[n]==(int)route,"route/count parity");
   ck(!memcmp(tokens+n,accepted,count*sizeof(int)),"routed token parity");
   ck(!memcmp(logits+(size_t)n*DS4_N_VOCAB,s->logits,DS4_N_VOCAB*sizeof(float)),"routed full-logit parity");
  }else{counts[n]=count;routes[n]=route;memcpy(tokens+n,accepted,count*sizeof(int));memcpy(logits+(size_t)n*DS4_N_VOCAB,s->logits,DS4_N_VOCAB*sizeof(float));}
  n+=count;
 }
 if(!compare)printf("HOT_ROUTE_COUNTS plain=%d/%d lookup=%d/%d neural=%d/%d\n",calls[0],commits[0],calls[1],commits[1],calls[2],commits[2]);
}
int main(int argc,char **argv){
 ck(argc==2,"model");ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,.context_size=8192,.power_percent=100,.warm_weights=true};
 if(getenv("HOT_LONG"))opt.context_size=16384;
 if(getenv("HOT_DSPARK")){opt.dspark=true;opt.dspark_exact_sampling=true;opt.mtp_path=getenv("HOT_DSPARK");}
 ds4_engine *e=NULL;ck(ds4_engine_open(&e,&opt)==0,"engine");char err[512]={0};
 ds4_tokens text={0};ds4_tokenize_text(e,"Preserve the exact cached prompt and its attention state.\n",&text);
 int lengths[]={17,127,128,129,259,2048};
 if(getenv("HOT_LONG"))lengths[5]=8192;
 for(int k=getenv("HOT_SINGLE")?5:0;k<6;k++){
  ds4_tokens p={0};for(int j=0;j<lengths[k];j++)token_vec_push(&p,text.v[j%text.len]);
  if(getenv("HOT_TASK")){ds4_tokens_free(&p);ds4_encode_chat_prompt(e,NULL,getenv("HOT_TASK"),DS4_THINK_NONE,&p);}
  ds4_session *s=NULL;ck(ds4_session_create(&s,e,opt.context_size)==0,"session");ck(ds4_session_sync(s,&p,err,sizeof(err))==0,err);
  ds4_session_snapshot initial={0};ck(ds4_session_save_snapshot(s,&initial,err,sizeof(err))==0,err);
  double start=now_sec();ck(ds4_session_mark_rewind_point(s),"mark");double mark=now_sec()-start;
  int tokens[128],counts[128]={0},routes[128]={0};
  float *routed_logits=getenv("HOT_ROUTE")?malloc((size_t)128*DS4_N_VOCAB*sizeof(float)):NULL;
  ck(!getenv("HOT_ROUTE")||routed_logits,"route oracle allocation");
  if(routed_logits)route_run(s,tokens,counts,routes,routed_logits,false);
  else for(int j=0;j<128;j++){tokens[j]=ds4_session_argmax(s);ck(ds4_session_eval(s,tokens[j],err,sizeof(err))==0,err);}
  start=now_sec();ds4_session_rewind(s,p.len);double restore=now_sec()-start;ck(ds4_session_checkpoint_valid(s),"valid restore");same_snapshot(s,&initial);
  if(routed_logits)route_run(s,tokens,counts,routes,routed_logits,true);
  else for(int j=0;j<128;j++){ck(tokens[j]==ds4_session_argmax(s),"regeneration tokens");ck(ds4_session_eval(s,tokens[j],err,sizeof(err))==0,err);}
  ds4_session_rewind(s,p.len);same_snapshot(s,&initial);
  // Extend far enough to overwrite the complete physical raw ring, using
  // prefill rather than thousands of unnecessary one-token decode calls.
  ds4_tokens extended={0};ds4_tokens_copy(&extended,&p);
  for(uint32_t j=0;j<s->graph.raw_cap+8;j++)token_vec_push(&extended,text.v[j%text.len]);
  ck(extended.len<opt.context_size,"wrap capacity");ck(ds4_session_sync(s,&extended,err,sizeof(err))==0,err);
  ds4_session_rewind(s,p.len);same_snapshot(s,&initial);
  if(routed_logits)route_run(s,tokens,counts,routes,routed_logits,true);
  else for(int j=0;j<16;j++){ck(tokens[j]==ds4_session_argmax(s),"post-wrap regeneration tokens");ck(ds4_session_eval(s,tokens[j],err,sizeof(err))==0,err);}
  ds4_session_rewind(s,p.len);same_snapshot(s,&initial);
  ds4_session_rewind(s,p.len-1);ck(!ds4_session_checkpoint_valid(s),"unmarked rewind must invalidate");
  ck(ds4_session_sync(s,&p,err,sizeof(err))==0,err);same_snapshot(s,&initial);
  ck(ds4_session_mark_rewind_point(s),"remark");ck(ds4_session_load_snapshot(s,&initial,err,sizeof(err))==0,err);
  ck(!ds4_session_can_rewind_to(s,p.len),"payload load invalidates stale mark");
  printf("HOT_PASS length=%d raw_cap=%u mark_ms=%.3f restore_ms=%.3f extra_bytes=%llu\n",p.len,s->graph.raw_cap,mark*1000,restore*1000,(unsigned long long)ds4_gpu_tensor_bytes(s->ds4_rewind_state));fflush(stdout);
  free(routed_logits);ds4_session_snapshot_free(&initial);ds4_tokens_free(&extended);ds4_tokens_free(&p);ds4_session_free(s);
 }
 ds4_tokens_free(&text);ds4_engine_close(e);printf("HOT_COMPLETE pass=%d\n",getenv("HOT_SINGLE")?1:6);return 0;
}
