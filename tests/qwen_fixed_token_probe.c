/* Public-API probe, compiled unchanged against both engine revisions. Token
 * IDs are shared, bypassing the older reference's different chat tokenizer. */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void ck(int ok,const char *msg){if(!ok){fprintf(stderr,"FIXED_PROBE_FAIL %s\n",msg);exit(2);}}
int main(int argc,char **argv){
 ck(argc==5,"model token-file logit-file write/read");
 int write_tokens=!strcmp(argv[4],"write");char err[512]={0};
 ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,.context_size=4096,.power_percent=100};
 ds4_engine *e=NULL;ck(!ds4_engine_open(&e,&opt),"engine");
 ds4_tokens p={0};FILE *ids=fopen(argv[2],write_tokens?"wb":"rb");ck(ids!=NULL,"token file");
 if(write_tokens){
  ds4_tokenize_rendered_chat(e,"<|im_start|>user\nWrite a Python function to merge two sorted lists. Explain its time complexity.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",&p);
  ck(fwrite(&p.len,4,1,ids)==1&&fwrite(p.v,4,p.len,ids)==(size_t)p.len,"write prompt");
 }else{
  ck(fread(&p.len,4,1,ids)==1&&p.len>0&&p.len<512,"read prompt length");
  p.cap=p.len;p.v=malloc(p.len*4);ck(p.v&&fread(p.v,4,p.len,ids)==(size_t)p.len,"read prompt");
 }
 ds4_session *s=NULL;ck(!ds4_session_create(&s,e,4096),"session");ck(!ds4_session_sync(s,&p,err,sizeof(err)),err);
 int vocab=ds4_engine_vocab_size(e);float *logits=malloc(vocab*4ull);FILE *out=fopen(argv[3],"wb");ck(logits&&out,"output");
 ck(fwrite(&vocab,4,1,out)==1,"header");
 for(int i=0;i<32;i++){
  ck(ds4_session_copy_logits(s,logits,vocab)==vocab,"copy logits");
  ck(fwrite(logits,4,vocab,out)==(size_t)vocab,"write logits");
  int token=ds4_session_argmax(s);
  if(write_tokens)ck(fwrite(&token,4,1,ids)==1,"write token");
  else ck(fread(&token,4,1,ids)==1,"read token");
  ck(!ds4_session_eval(s,token,err,sizeof(err)),err);
 }
 fclose(out);fclose(ids);free(logits);ds4_tokens_free(&p);ds4_session_free(s);ds4_engine_close(e);
 puts("FIXED_PROBE_PASS rows=32");return 0;
}
