// Production-route repeatability and timing harness. Two warmups, then ABBA.
// A no-op flag means A/A: timing differences are noise, not a speedup.
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void check(int ok,const char *err) { if(!ok) { fprintf(stderr,"FAIL %s\n",err); exit(2); } }
static char *read_text(const char *path) { FILE *f=fopen(path,"rb"); check(f!=NULL,path); fseek(f,0,SEEK_END); long n=ftell(f); rewind(f); char *s=malloc(n+1); check(s!=NULL,"alloc"); check(fread(s,1,n,f)==(size_t)n,"read"); fclose(f); s[n]=0; return s; }
int main(int argc,char **argv) {
    if(argc<7) { fprintf(stderr,"usage: model flag temp tokens ctx prompt...\n"); return 2; }
    const float temp=(float)atof(argv[3]); const int limit=atoi(argv[4]),ctx=atoi(argv[5]);
    ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,
        .context_size=ctx,.power_percent=100,.warm_weights=true,
        .glm_mtp=getenv("WIDE_MTP")!=NULL,.glm_mtp_timing=getenv("WIDE_MTP_TIMING")!=NULL};
    ds4_engine *e=NULL; check(ds4_engine_open(&e,&opt)==0,"engine");
    int vocab=ds4_engine_vocab_size(e),eos=ds4_token_eos(e);
    char err[512]={0};
    for(int file=6;file<argc;file++) {
        char *text=read_text(argv[file]); ds4_tokens prompt={0};
        ds4_encode_chat_prompt(e,NULL,text,DS4_THINK_NONE,&prompt); free(text);
        check(prompt.len+limit<ctx,"prompt capacity");
        int *reference=calloc(limit,sizeof(int)),reference_n=0;
        float *reference_logits=malloc((size_t)vocab*4),*logits=malloc((size_t)vocab*4);
        float *reference_prefill=malloc((size_t)vocab*4);
        const int order[6]={-1,-2,0,1,1,0};
        for(int iteration=0;iteration<6;iteration++) {
            const int run=iteration==0?0:iteration==1?-1:iteration-1;
            const int mode=order[iteration];
            unsetenv(argv[2]);
            ds4_session *s=NULL; check(ds4_session_create(&s,e,ctx)==0,"session");
            double p0=now(); check(ds4_session_sync(s,&prompt,err,sizeof(err))==0,err); double pp=now()-p0;
            check(ds4_session_copy_logits(s,logits,vocab)==vocab,"prefill logits");
            if(run==1) memcpy(reference_prefill,logits,(size_t)vocab*4);
            printf("prefill_check run=%d exact=%d\n",run,run<=1 || !memcmp(reference_prefill,logits,(size_t)vocab*4));
            if(mode==1 || mode==-2) setenv(argv[2],"1",1);
            int *output=calloc(limit,sizeof(int)),n=0,cycles[3]={0},commits[3]={0};
            double costs[3]={0}; uint64_t rng=1234567;
            double g0=now();
            while(n<limit) {
                double t=now(); int token=ds4_session_sample(s,temp,40,.95f,.05f,&rng);
                if(ds4_token_is_stop(e,token)) break;
                int accepted[32],count=1;
                ds4_decode_route route=ds4_session_select_decode_route(s,token,true,opt.glm_mtp,false);
                if(route==DS4_DECODE_ROUTE_PROMPT_LOOKUP) {
                    count=temp>0 ? ds4_session_eval_prompt_lookup_sampled(s,token,limit-n,eos,temp,40,.95f,.05f,&rng,accepted,32,err,sizeof(err)) :
                        ds4_session_eval_prompt_lookup_argmax(s,token,limit-n,eos,accepted,32,err,sizeof(err));
                } else if(route==DS4_DECODE_ROUTE_NEURAL_SPECULATION) {
                    count=ds4_session_eval_speculative(s,token,limit-n,eos,temp,40,.95f,.05f,&rng,accepted,32,err,sizeof(err));
                } else { check(ds4_session_eval(s,token,err,sizeof(err))==0,err); accepted[0]=token; }
                check(count>0&&count<=32&&count<=limit-n,err);
                const double dt=now()-t;
                cycles[route]++; commits[route]+=count; costs[route]+=dt;
                for(int i=0;i<count;i++) output[n++]=accepted[i];
            }
            double gen=now()-g0;
            check(ds4_session_copy_logits(s,logits,vocab)==vocab,"logits");
            const char *save=getenv("WIDE_SAVE_LOGITS");
            if(save && run==1) {
                char path[1024]; snprintf(path,sizeof(path),"%s-%d.bin",save,file-6);
                FILE *f=fopen(path,"wb"); check(f!=NULL,"save logits");
                check(fwrite(logits,4,vocab,f)==(size_t)vocab,"save logits write"); fclose(f);
            }
            int exact_tokens=1,exact_logits=1;
            if(run==1) { memcpy(reference,output,n*sizeof(int)); reference_n=n; memcpy(reference_logits,logits,vocab*4); }
            if(run>1) {
                exact_tokens=n==reference_n&&!memcmp(reference,output,n*sizeof(int));
                exact_logits=!memcmp(reference_logits,logits,vocab*4);
                if(!exact_logits) {
                    size_t count=0; float maxerr=0;
                    for(int i=0;i<vocab;i++) if(memcmp(reference_logits+i,logits+i,4)) {
                        if(!count) fprintf(stderr,"first_logit_difference id=%d base=%a candidate=%a\n",i,reference_logits[i],logits[i]);
                        count++; maxerr=fmaxf(maxerr,fabsf(reference_logits[i]-logits[i]));
                    }
                    fprintf(stderr,"logit_differences=%zu maxerr=%.9g\n",count,maxerr);
                }
            }
            printf("case=%s run=%d candidate=%d temp=%.2f input=%d output=%d prefill_s=%.6f gen_s=%.6f tps=%.4f token_exact=%d logits_exact=%d plain_ms=%.3f lookup_ms=%.3f mtp_ms=%.3f plain_n=%d lookup_n=%d mtp_n=%d\n",
                argv[file],run,mode,temp,prompt.len,n,pp,gen,n/gen,exact_tokens,exact_logits,
                costs[0]*1000,costs[1]*1000,costs[2]*1000,commits[0],commits[1],commits[2]); fflush(stdout);
            free(output); ds4_session_free(s);
            if(run>1&&(!exact_tokens||!exact_logits)) {
                fprintf(stderr,"PARITY FAILURE\n");
                if(!getenv("WIDE_DIAGNOSTIC")) return 3;
            }
        }
        free(reference_prefill); free(logits); free(reference_logits); free(reference); ds4_tokens_free(&prompt);
    }
    ds4_engine_close(e); return 0;
}
