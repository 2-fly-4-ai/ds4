#define main original_qwen_kernel_main
#include "test_qwen4_kernels.c"
#undef main
int main(void) {
    arena_t a={0};a.size=64ull<<20;a.base=mmap(NULL,a.size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    require_ok(a.base!=MAP_FAILED&&ds4_gpu_init()&&ds4_gpu_set_model_map(a.base,a.size),"norm setup");
    const uint32_t types[]={0,1,8}, sizes[]={2048,8192};
    for(int ty=0;ty<3;ty++)for(int sz=0;sz<2;sz++) {
        const uint32_t E=2560,hc=4,T=sizes[sz],type=types[ty],dim=E*hc;
        const uint64_t n=(uint64_t)T*dim, ni=(uint64_t)T*hc*DS4_QWEN4_HC_CHUNKS*4;
        double *shadow=NULL;uint64_t gamma=arena_f32(&a,dim,&shadow,.5,1.5);free(shadow);
        uint64_t inj=type==8?arena_q8_0(&a,4,dim,&shadow,.05):type==1?arena_f16(&a,4ull*dim,&shadow,.05):arena_f32(&a,4ull*dim,&shadow,-.05,.05);free(shadow);
        float *input=rand_vec(n,1),*ref=malloc(n*4),*got=malloc(n*4),*ri=malloc(ni*4),*gi=malloc(ni*4);
        require_ok(input&&ref&&got&&ri&&gi,"norm allocation");
        ds4_gpu_tensor *R=upload(input,n),*x=upload(NULL,n),*p=upload(NULL,ni);
        const int order[]={0,1,1,0,1,0,0,1};
        for(int rep=0;rep<8;rep++) {
            int mode=order[rep];if(mode)unsetenv("DS4_QWEN_NORM_REUSE_DISABLE");else setenv("DS4_QWEN_NORM_REUSE_DISABLE","1",1);
            double t=bench_now();require_ok(ds4_gpu_begin_commands(),"norm begin");
            for(int j=0;j<8;j++)require_ok(ds4_gpu_qwen4_hc_norm_tensor(x,p,R,a.base,a.size,gamma,inj,type,T,E,hc,4,1e-6),"norm dispatch");
            require_ok(ds4_gpu_end_commands(),"norm end");double ms=(bench_now()-t)*1000/8;
            require_ok(ds4_gpu_tensor_read(x,0,mode?got:ref,n*4)&&ds4_gpu_tensor_read(p,0,mode?gi:ri,ni*4),"norm read");
            if(rep)require_ok(!memcmp(ref,got,n*4)&&!memcmp(ri,gi,ni*4),"norm exact output/injection");
            printf("QWEN_NORM type=%u T=%u mode=%d repeat=%d ms=%.5f exact=1\n",type,T,mode,rep,ms);fflush(stdout);
        }
        free(input);free(ref);free(got);free(ri);free(gi);ds4_gpu_tensor_free(R);ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(p);
    }
    puts("QWEN_NORM_PORT_PASS");return 0;
}
