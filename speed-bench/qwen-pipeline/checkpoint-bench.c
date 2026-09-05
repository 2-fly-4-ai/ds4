#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static char err[512];
#define OK(x) do { if ((x) != 0) { fprintf(stderr, "FAIL %s: %s\n", #x, err); exit(1); } } while(0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void decode(ds4_session *s, int *ids) { for(int i=0;i<32;i++) {ids[i]=ds4_session_argmax(s); OK(ds4_session_eval(s,ids[i],err,sizeof(err)));} }
int main(int argc,char **argv) {
 if(argc!=3) return 2;
 ds4_engine_options opt={0};opt.model_path=argv[1];opt.backend=DS4_BACKEND_METAL;opt.n_threads=8;
 ds4_engine *e=NULL;OK(ds4_engine_open(&e,&opt));
 FILE *f=fopen(argv[2],"rb");if(!f)return 2;fseek(f,0,SEEK_END);long len=ftell(f);rewind(f);char *text=calloc(len+1,1);fread(text,1,len,f);fclose(f);
 ds4_tokens full={0};ds4_encode_chat_prompt(e,NULL,text,DS4_THINK_NONE,&full);free(text);
 ds4_session *s=NULL;OK(ds4_session_create(&s,e,32768));
 const int lengths[]={2048,8192,30000};
 printf("tokens,prefill_ms,save_ms,snapshot_MiB,restore1_ms,restore2_ms,restore3_ms,parity\n");fflush(stdout);
 for(int j=0;j<3;j++) {
  ds4_tokens p={0};for(int k=0;k<lengths[j]&&k<full.len;k++)ds4_tokens_push(&p,full.v[k]);
  ds4_session_invalidate(s);
  double t=now();OK(ds4_session_sync(s,&p,err,sizeof(err)));double pp=now()-t;
  ds4_session_snapshot snap={0};t=now();OK(ds4_session_save_snapshot(s,&snap,err,sizeof(err)));double save=now()-t;
  int original[32],reloaded[32];decode(s,original);
  double restores[3];int same=1;
  for(int rep=0;rep<3;rep++) {
   t=now();OK(ds4_session_load_snapshot(s,&snap,err,sizeof(err)));restores[rep]=now()-t;
   decode(s,reloaded);if(memcmp(original,reloaded,sizeof(original)))same=0;
  }
  printf("%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n",p.len,1000*pp,1000*save,snap.len/1048576.0,1000*restores[0],1000*restores[1],1000*restores[2],same?"PASS":"FAIL");fflush(stdout);
  ds4_session_snapshot_free(&snap);ds4_tokens_free(&p);if(!same)return 1;
 }
 ds4_session_free(s);ds4_tokens_free(&full);ds4_engine_close(e);return 0;
}
