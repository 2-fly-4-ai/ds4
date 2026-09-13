/* Model-shaped GPU math versus the existing independent double oracle.
 * Sixteen synthetic experts bound CPU reference memory; routing separately
 * exercises the actual 256-expert top-8 shape. */
#define main unused_qwen4_kernel_main
#include "test_qwen4_kernels.c"
#undef main
int main(void) {
    arena_t a={0};a.size=256ull<<20;
    a.base=mmap(NULL,a.size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    require_ok(a.base!=MAP_FAILED&&ds4_gpu_init()&&ds4_gpu_set_model_map(a.base,a.size),"setup");
    test_router(&a,256,8,1);
    test_router(&a,256,8,5);
    test_moe_types(&a,16,8,2048,512,1,8u,8u);
    test_moe_types(&a,16,8,2048,512,5,8u,8u);
    puts("QWEN35_MOE_ORACLE_PASS");
    return 0;
}
