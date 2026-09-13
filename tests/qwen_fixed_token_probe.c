/* Public-API probe, compiled unchanged against both engine revisions. Token
 * IDs are shared, bypassing the older reference's different chat tokenizer. */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void ck(int ok,const char *msg){if(!ok){fprintf(stderr,"FIXED_PROBE_FAIL %s\n",msg);exit(2);}}
int main(int argc,char **argv){
 ck(argc==5,"model token-file logit-file write/read");
 int write_tokens=!strcmp(argv[4],"write");char err[512]={0};
 ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,.context_size=4096,.power_percent=100};
 ds4_engine *e=NULL;ck(!ds4_engine_open(&e,&opt),"engine");
 ds4_tokens p={0};FILE *ids=fopen(argv[2],write_tokens?"wb":"rb");ck(ids!=NULL,"token file");
 if(write_tokens){
  ds4_tokenize_rendered_chat(e,"<|im_start|>user\nWrite a Python function to merge two sorted lists. Explain its time complexity.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",&p);
  int length=getenv("PROBE_CONTEXT")?atoi(getenv("PROBE_CONTEXT")):0;
  if(length>p.len&&length<4000){int old=p.len;p.v=realloc(p.v,length*4);ck(p.v!=NULL,"extend prompt");for(int j=old;j<length;j++)p.v[j]=p.v[j%old];p.len=p.cap=length;}
  ck(fwrite(&p.len,4,1,ids)==1&&fwrite(p.v,4,p.len,ids)==(size_t)p.len,"write prompt");
 }else{
  ck(fread(&p.len,4,1,ids)==1&&p.len>0&&p.len<4000,"read prompt length");
  p.cap=p.len;p.v=malloc(p.len*4);ck(p.v&&fread(p.v,4,p.len,ids)==(size_t)p.len,"read prompt");
 }
 ds4_session *s=NULL;ck(!ds4_session_create(&s,e,4096),"session");double start=seconds();ck(!ds4_session_sync(s,&p,err,sizeof(err)),err);double prefill=seconds()-start,decode=0;
 int vocab=ds4_engine_vocab_size(e);float *logits=malloc(vocab*4ull);FILE *out=fopen(argv[3],"wb");ck(logits&&out,"output");
 ck(fwrite(&vocab,4,1,out)==1,"header");
 for(int i=0;i<32;i++){
  ck(ds4_session_copy_logits(s,logits,vocab)==vocab,"copy logits");
  ck(fwrite(logits,4,vocab,out)==(size_t)vocab,"write logits");
  int token=ds4_session_argmax(s);
  if(write_tokens)ck(fwrite(&token,4,1,ids)==1,"write token");
  else ck(fread(&token,4,1,ids)==1,"read token");
  start=seconds();ck(!ds4_session_eval(s,token,err,sizeof(err)),err);decode+=seconds()-start;
 }
 fclose(out);fclose(ids);free(logits);ds4_tokens_free(&p);ds4_session_free(s);ds4_engine_close(e);
 printf("FIXED_PROBE_PASS rows=32 prefill_s=%.6f decode_s=%.6f decode_tps=%.3f\n",prefill,decode,32/decode);return 0;
}
