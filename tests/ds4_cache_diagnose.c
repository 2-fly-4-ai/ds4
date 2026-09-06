#include "../ds4.c"
static void ck(bool ok,const char *why){if(!ok){fprintf(stderr,"FAIL %s\n",why);exit(2);}}
static void compare(ds4_session *s,const float *ref,const char *name){
 int diff=0;float mx=0;for(uint32_t i=0;i<DS4_N_VOCAB;i++){diff+=memcmp(ref+i,s->logits+i,4)!=0;float d=fabsf(ref[i]-s->logits[i]);if(d>mx)mx=d;}
 printf("COMPARE %s different=%d max_abs=%.8g top=%d ncomp3=%u index3=%u\n",name,diff,mx,ds4_session_argmax(s),s->graph.layer_n_comp[3],s->graph.layer_n_index_comp[3]);fflush(stdout);
}
int main(int argc,char **argv){
 ck(argc==2,"model");ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,.context_size=4096,.power_percent=100,.warm_weights=true};
 ds4_engine *e=NULL;ck(ds4_engine_open(&e,&opt)==0,"engine");ds4_session *s=NULL;ck(ds4_session_create(&s,e,4096)==0,"session");
 ds4_tokens p={0};ds4_encode_chat_prompt(e,NULL,"Write a Python function to add two integers. Briefly explain it.",DS4_THINK_NONE,&p);char err[512]={0};
 ck(ds4_session_sync(s,&p,err,sizeof(err))==0,err);float *ref=malloc(DS4_N_VOCAB*4);memcpy(ref,s->logits,DS4_N_VOCAB*4);
 ds4_session_snapshot snap={0};ck(ds4_session_save_snapshot(s,&snap,err,sizeof(err))==0,err);printf("PROMPT len=%d raw_cap=%u snapshot_bytes=%llu\n",p.len,s->graph.raw_cap,(unsigned long long)snap.len);compare(s,ref,"cold");
 ds4_session_rewind(s,p.len-1);ck(ds4_session_sync(s,&p,err,sizeof(err))==0,err);compare(s,ref,"rewind_without_generation");
 ck(ds4_session_load_snapshot(s,&snap,err,sizeof(err))==0,err);
 for(int i=0;i<128;i++)ck(ds4_session_eval(s,ds4_session_argmax(s),err,sizeof(err))==0,err);
 compare(s,ref,"generated128");ds4_session_rewind(s,p.len-1);ck(ds4_session_sync(s,&p,err,sizeof(err))==0,err);compare(s,ref,"rewind_after_generation");
 ck(ds4_session_load_snapshot(s,&snap,err,sizeof(err))==0,err);compare(s,ref,"full_snapshot_restored");
 ds4_session_snapshot_free(&snap);free(ref);ds4_tokens_free(&p);ds4_session_free(s);ds4_engine_close(e);return 0;
}
