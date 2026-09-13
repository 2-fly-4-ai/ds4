/* Bounded synthetic test; no model or large CPU inference. */
#define main unused_qwen4_kernel_main
#include "test_qwen4_kernels.c"
#undef main

static void small_qwen_reference(uint32_t Hk, uint32_t Hv, uint32_t D, uint32_t K, uint32_t T,
                          const float *qkv_in, const float *z, const float *ab_in,
                          const double *conv_w, const double *ssm_a, const double *dt, const double *norm_w,
                          double *state, double *hist, double *out) {
    const uint32_t kd = Hk * D, vd = Hv * D, C = 2 * kd + vd;
    double *conv = malloc(C * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t c = 0; c < C; c++) {
            double acc = conv_w[c * K + (K - 1)] * qkv_in[t * C + c];
            for (uint32_t k = 0; k + 1 < K; k++) acc += conv_w[c * K + k] * hist[k * C + c];
            conv[c] = silu_d(acc);
        }
        for (uint32_t k = 0; k + 2 < K; k++) memcpy(hist + k * C, hist + (k + 1) * C, C * sizeof(double));
        for (uint32_t c = 0; c < C; c++) hist[(K - 2) * C + c] = qkv_in[t * C + c];
        for (uint32_t h = 0; h < Hk; h++) {
            double sq = 0.0, sk = 0.0;
            for (uint32_t i = 0; i < D; i++) { sq += conv[h * D + i] * conv[h * D + i]; sk += conv[kd + h * D + i] * conv[kd + h * D + i]; }
            const double qs = 1.0 / fmax(sqrt(sq), 1e-6) / sqrt((double)D), ks = 1.0 / fmax(sqrt(sk), 1e-6);
            for (uint32_t i = 0; i < D; i++) { conv[h * D + i] *= qs; conv[kd + h * D + i] *= ks; }
        }
        for (uint32_t j = 0; j < Hv; j++) {
            const uint32_t kh = j % Hk;
            const double g = exp(ssm_a[j] * softplus_d((double)ab_in[t * 2 * Hv + j] + dt[j]));
            const double beta = sigmoid_d(ab_in[t * 2 * Hv + Hv + j]);
            double *S = state + (uint64_t)j * D * D;   /* [dv][dk] */
            const double *q = conv + kh * D, *k = conv + kd + kh * D, *v = conv + 2 * kd + j * D;
            double o[128];
            for (uint32_t dv = 0; dv < D; dv++) {
                double u = 0.0;
                for (uint32_t dk = 0; dk < D; dk++) { S[dv * D + dk] *= g; u += S[dv * D + dk] * k[dk]; }
                const double delta = (v[dv] - u) * beta;
                double acc = 0.0;
                for (uint32_t dk = 0; dk < D; dk++) { S[dv * D + dk] += k[dk] * delta; acc += S[dv * D + dk] * q[dk]; }
                o[dv] = acc;
            }
            double ss = 0.0;
            for (uint32_t dv = 0; dv < D; dv++) ss += o[dv] * o[dv];
            const double r = 1.0 / sqrt(ss / D + 1e-6);
            for (uint32_t dv = 0; dv < D; dv++)
                out[((uint64_t)t * Hv + j) * D + dv] = o[dv] * r * norm_w[dv] * silu_d(z[((uint64_t)t * Hv + j) * D + dv]);
        }
    }
    free(conv);
}

static void test_small_gdn(arena_t *a, bool moe, uint32_t T, uint32_t layer, bool warm) {
    const uint32_t L = moe ? 40 : 64, H = moe ? 32 : 48;
    const uint32_t D = 128, C = 4096 + H * D, Z = H * D;
    const uint64_t CN = (uint64_t)L*C*4, SN = (uint64_t)L*H*D*D;
    double *cw, *aw, *dw, *nw;
    uint64_t co=arena_f32(a,C*4,&cw,-.5,.5);
    uint64_t ao=arena_f32(a,H,&aw,-8,-.1);
    uint64_t dto=arena_f32(a,H,&dw,.2,1.5);
    uint64_t no=arena_f32(a,D,&nw,.8,1.2);
    float *qkv=rand_vec((uint64_t)T*C,1), *z=rand_vec((uint64_t)T*Z,1);
    float *av=rand_vec((uint64_t)T*H,1), *bv=rand_vec((uint64_t)T*H,1);
    float *ab=malloc((uint64_t)T*H*2*4);
    for(uint32_t t=0;t<T;t++) {
        memcpy(ab+(uint64_t)t*H*2,av+(uint64_t)t*H,H*4);
        memcpy(ab+(uint64_t)t*H*2+H,bv+(uint64_t)t*H,H*4);
    }
    double *rs=calloc((uint64_t)H*D*D,sizeof(double));
    double *rh=calloc((uint64_t)C*3,sizeof(double));
    double *ro=calloc((uint64_t)T*Z,sizeof(double));
    require_ok(ab&&rs&&rh&&ro,"reference allocation");
    float *initial_state=rand_vec((uint64_t)H*D*D,warm?.1f:0.0f);
    float *initial_conv=rand_vec((uint64_t)C*4,warm?1.0f:0.0f);
    for(uint64_t i=0;i<(uint64_t)H*D*D;i++)rs[i]=initial_state[i];
    for(uint32_t c=0;c<C;c++)for(uint32_t k=0;k<3;k++)rh[(uint64_t)k*C+c]=initial_conv[c*4+k+1];
    small_qwen_reference(16,H,D,4,T,qkv,z,ab,cw,aw,dw,nw,rs,rh,ro);
    const bool snapshots=moe && T>1 && T<=8;
    ds4_gpu_tensor *cs=snapshots?upload(NULL,(uint64_t)T*CN):NULL;
    ds4_gpu_tensor *ss=snapshots?upload(NULL,(uint64_t)T*SN):NULL;
    float *batch_out=NULL, *batch_state=NULL, *batch_conv=NULL;
    for(int mode=0;mode<2;mode++) {
        ds4_gpu_tensor *Q=upload(qkv,(uint64_t)T*C), *Zt=upload(z,(uint64_t)T*Z);
        ds4_gpu_tensor *A=upload(av,(uint64_t)T*H), *B=upload(bv,(uint64_t)T*H);
        ds4_gpu_tensor *O=upload(NULL,(uint64_t)T*Z), *S=upload(NULL,SN), *Conv=upload(NULL,CN);
        require_ok(ds4_gpu_tensor_write(S,(uint64_t)layer*H*D*D*4,initial_state,(uint64_t)H*D*D*4)&&
                   ds4_gpu_tensor_write(Conv,(uint64_t)layer*C*4*4,initial_conv,(uint64_t)C*4*4),"initial recurrent state");
        const uint32_t step=mode?1:T;
        for(uint32_t t=0;t<T;t+=step) {
            ds4_gpu_tensor *q=ds4_gpu_tensor_view(Q,(uint64_t)t*C*4,(uint64_t)step*C*4);
            ds4_gpu_tensor *zt=ds4_gpu_tensor_view(Zt,(uint64_t)t*Z*4,(uint64_t)step*Z*4);
            ds4_gpu_tensor *at=ds4_gpu_tensor_view(A,(uint64_t)t*H*4,(uint64_t)step*H*4);
            ds4_gpu_tensor *bt=ds4_gpu_tensor_view(B,(uint64_t)t*H*4,(uint64_t)step*H*4);
            ds4_gpu_tensor *ot=ds4_gpu_tensor_view(O,(uint64_t)t*Z*4,(uint64_t)step*Z*4);
            if(moe) {
                require_ok(!ds4_gpu_qwen35_gdn_core_rows_tensor(ot,Conv,S,q,zt,at,bt,
                    a->base,a->size,co,ao,dto,no,40,step,false,NULL,NULL),"reject bad layer");
                require_ok(!ds4_gpu_qwen35_gdn_core_rows_tensor(ot,Conv,S,q,zt,at,bt,
                    a->base,a->size,co,ao,dto,no,layer,step,true,NULL,NULL),"reject absent snapshots");
                require_ok(!ds4_gpu_qwen35_gdn_core_rows_tensor(at,Conv,S,q,zt,at,bt,
                    a->base,a->size,co,ao,dto,no,layer,step,false,NULL,NULL),"reject undersized output");
                require_ok(ds4_gpu_qwen35_gdn_core_rows_tensor(ot,Conv,S,q,zt,at,bt,
                    a->base,a->size,co,ao,dto,no,layer,step,!mode&&snapshots,cs,ss),"35B GDN");
            } else {
                require_ok(ds4_gpu_qwen_gdn_core_rows_tensor(ot,Conv,S,q,zt,at,bt,
                    a->base,a->size,co,ao,dto,no,layer,step,false),"27B GDN");
            }
            if(mode&&snapshots) {
                ds4_gpu_tensor *sc=ds4_gpu_tensor_view(cs,((uint64_t)t*CN+(uint64_t)layer*C*4)*4,(uint64_t)C*4*4);
                ds4_gpu_tensor *st=ds4_gpu_tensor_view(ss,((uint64_t)t*SN+(uint64_t)layer*H*D*D)*4,(uint64_t)H*D*D*4);
                ds4_gpu_tensor *lc=ds4_gpu_tensor_view(Conv,(uint64_t)layer*C*4*4,(uint64_t)C*4*4);
                ds4_gpu_tensor *ls=ds4_gpu_tensor_view(S,(uint64_t)layer*H*D*D*4,(uint64_t)H*D*D*4);
                float *c1=download(sc,C*4),*c2=download(lc,C*4);
                float *s1=download(st,(uint64_t)H*D*D),*s2=download(ls,(uint64_t)H*D*D);
                require_ok(!memcmp(c1,c2,C*4*4)&&!memcmp(s1,s2,(uint64_t)H*D*D*4),"every prefix snapshot exact");
                free(c1);free(c2);free(s1);free(s2);
                ds4_gpu_tensor_free(sc);ds4_gpu_tensor_free(st);ds4_gpu_tensor_free(lc);ds4_gpu_tensor_free(ls);
            }
            ds4_gpu_tensor_free(q);ds4_gpu_tensor_free(zt);ds4_gpu_tensor_free(at);
            ds4_gpu_tensor_free(bt);ds4_gpu_tensor_free(ot);
        }
        float *out=download(O,(uint64_t)T*Z), *state=download(S,SN), *conv=download(Conv,CN);
        check_close("small GDN double reference output",out,ro,(uint64_t)T*Z,5e-5);
        check_close("small GDN double reference state",state+(uint64_t)layer*H*D*D,rs,(uint64_t)H*D*D,5e-5);
        for(uint64_t i=0;i<SN;i++) if(i<(uint64_t)layer*H*D*D || i>=(uint64_t)(layer+1)*H*D*D)
            require_ok(state[i]==0,"neighbor state unchanged");
        for(uint64_t i=0;i<CN;i++) if(i<(uint64_t)layer*C*4 || i>=(uint64_t)(layer+1)*C*4)
            require_ok(conv[i]==0,"neighbor conv unchanged");
        if(!mode){batch_out=out;batch_state=state;batch_conv=conv;}
        else {
            require_ok(!memcmp(batch_out,out,(uint64_t)T*Z*4)&&
                       !memcmp(batch_state,state,SN*4)&&!memcmp(batch_conv,conv,CN*4),
                       "batch/scalar output/state/conv exact");
            free(out);free(state);free(conv);
        }
        ds4_gpu_tensor_free(Q);ds4_gpu_tensor_free(Zt);ds4_gpu_tensor_free(A);ds4_gpu_tensor_free(B);
        ds4_gpu_tensor_free(O);ds4_gpu_tensor_free(S);ds4_gpu_tensor_free(Conv);
    }
    printf("SMALL_GDN_PASS model=%s T=%u layer=%u warm=%u prefix_snapshots=%s\n",moe?"35B":"27B",T,layer,warm,snapshots?"exact":"not requested");
    fflush(stdout);
    free(batch_out);free(batch_state);free(batch_conv);
    if(cs)ds4_gpu_tensor_free(cs);if(ss)ds4_gpu_tensor_free(ss);
    free(cw);free(aw);free(dw);free(nw);free(qkv);free(z);free(av);free(bv);free(ab);free(rs);free(rh);free(ro);
    free(initial_state);free(initial_conv);
}

int main(void) {
    arena_t a={0};a.size=16ull<<20;
    a.base=mmap(NULL,a.size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    require_ok(a.base!=MAP_FAILED&&ds4_gpu_init()&&ds4_gpu_set_model_map(a.base,a.size),"setup");
    const uint32_t lengths[]={1,2,8,17};
    for(int warm=0;warm<2;warm++)for(int model=0;model<2;model++)for(int layer=0;layer<2;layer++)for(int i=0;i<4;i++)
        test_small_gdn(&a,model,lengths[i],layer?(model?39:63):0,warm);
    puts("SMALL_QWEN_GDN_SUITE_PASS");
    return 0;
}
