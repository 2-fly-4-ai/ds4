/* Inspect only the GGUF directory. Does not enable the unfinished runtime. */
#include "../ds4.c"
#include <assert.h>

static void shape(const ds4_model *m, uint32_t layer, const char *suffix,
                  uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    char name[128];
    snprintf(name,sizeof(name),"blk.%u.%s",layer,suffix);
    const ds4_tensor *t=required_tensor(m,name);
    if(t->ndim!=ndim || t->dim[0]!=d0 || (ndim>1 && t->dim[1]!=d1) ||
       (ndim>2 && t->dim[2]!=d2)) {
        fprintf(stderr,"35B contract mismatch: %s ndim=%u dims=%llu/%llu/%llu\n",
                name,t->ndim,(unsigned long long)t->dim[0],
                (unsigned long long)t->dim[1],(unsigned long long)t->dim[2]);
        exit(1);
    }
}

int main(int argc,char **argv) {
    assert(argc==2);
    ds4_model m={0};model_open(&m,argv[1],false,false);
    ds4_str arch={0};assert(model_get_string(&m,"general.architecture",&arch));
    assert(ds4_streq(arch,"qwen35moe"));
    uint32_t layers=0,width=0,experts=0,used=0;
    assert(model_get_u32(&m,"qwen35moe.block_count",&layers));
    assert(model_get_u32(&m,"qwen35moe.embedding_length",&width));
    assert(model_get_u32(&m,"qwen35moe.expert_count",&experts));
    assert(model_get_u32(&m,"qwen35moe.expert_used_count",&used));
    assert((layers==40 || layers==41)&&width==2048&&experts==256&&used==8);
    unsigned gdn=0,full=0;
    for(uint32_t l=0;l<40;l++) {
        shape(&m,l,"ffn_gate_inp.weight",2,2048,256,0);
        shape(&m,l,"ffn_gate_exps.weight",3,2048,512,256);
        shape(&m,l,"ffn_up_exps.weight",3,2048,512,256);
        shape(&m,l,"ffn_down_exps.weight",3,512,2048,256);
        if(tensor_by_namef(&m,"blk.%u.attn_qkv.weight",l)) {
            shape(&m,l,"attn_qkv.weight",2,2048,8192,0);
            shape(&m,l,"attn_gate.weight",2,2048,4096,0);
            shape(&m,l,"ssm_alpha.weight",2,2048,32,0);
            shape(&m,l,"ssm_beta.weight",2,2048,32,0);
            shape(&m,l,"ssm_out.weight",2,4096,2048,0);
            gdn++;
        } else {
            shape(&m,l,"attn_q.weight",2,2048,8192,0);
            shape(&m,l,"attn_k.weight",2,2048,512,0);
            shape(&m,l,"attn_v.weight",2,2048,512,0);
            shape(&m,l,"attn_output.weight",2,4096,2048,0);
            full++;
        }
    }
    assert(gdn==30&&full==10);
    const ds4_tensor *mtp=model_find_tensor(&m,"blk.40.nextn.eh_proj.weight");
    assert(mtp&&mtp->ndim==2&&mtp->dim[0]==4096&&mtp->dim[1]==2048);
    printf("35B_CONTRACT_PASS layers=%u width=%u experts=%u topk=%u GDN=%u full=%u native_MTP=present\n",
           layers,width,experts,used,gdn,full);
    model_close(&m);return 0;
}
