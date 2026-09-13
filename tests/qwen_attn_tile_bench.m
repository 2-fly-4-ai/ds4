#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
typedef struct {uint32_t nh,nk,dim,pos,layer,cap,gated,rows;} Args;
static void require(BOOL ok){if(!ok){fprintf(stderr,"FAIL\n");exit(2);}}
static id<MTLComputePipelineState> pipeline(id<MTLDevice>d,NSString*path,NSString*name){
 NSError *err=nil; NSString*s=[NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];require(s!=nil);
 MTLCompileOptions *opt=[MTLCompileOptions new];opt.mathMode=getenv("SAFE")?MTLMathModeSafe:MTLMathModeFast;
 id<MTLLibrary>lib=[d newLibraryWithSource:s options:opt error:&err];if(!lib)NSLog(@"%@",err);require(lib!=nil);
 id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];require(p!=nil);return p;
}
static double run(id<MTLCommandQueue>queue,id<MTLComputePipelineState>p,Args a,NSArray*buf){
 id<MTLCommandBuffer>cb=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];
 [enc setComputePipelineState:p];[enc setBytes:&a length:sizeof(a) atIndex:0];
 for(int i=0;i<5;i++)[enc setBuffer:buf[i] offset:0 atIndex:i+1];
 [enc dispatchThreadgroups:MTLSizeMake(a.nh,a.rows,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
 [enc endEncoding];[cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted);
 return (cb.GPUEndTime-cb.GPUStartTime)*1000;
}
int main(){@autoreleasepool{
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>queue=[d newCommandQueue];
 id<MTLComputePipelineState>b=pipeline(d,@"metal/qwen_gdn.metal",@"kernel_qwen_attn_decode_rows");
 id<MTLComputePipelineState>p=pipeline(d,@"metal/qwen_gdn.metal",@"kernel_qwen_attn_decode_tiled");
 puts("heads,context,rows,baseline_ms,tile_ms,speedup,max_abs,exact");
 for(int shape=0;shape<2;shape++)for(int c=0;c<7;c++)for(int r=0;r<4;r++){@autoreleasepool{
  int contexts[]={1,7,8,9,128,2048,3840}, rows[]={1,2,5,8};
  Args a={shape?24:16,shape?4:2,256,contexts[c]-1,1,4096,(r&1)?0:1,rows[r]};
  size_t n=a.nh*256*a.rows,kv=2*a.cap*a.nk*256;
  NSMutableArray*buf=[NSMutableArray new];size_t sizes[]={n,kv,kv,n,n};
  for(int i=0;i<5;i++){
   id<MTLBuffer>x=[d newBufferWithLength:sizes[i]*4 options:MTLResourceStorageModeShared];require(x!=nil);
   float *f=x.contents;uint32_t state=1234+i;
   for(size_t j=0;j<sizes[i];j++){state^=state<<13;state^=state>>17;state^=state<<5;f[j]=getenv("QWEN_TEST_RANDOM")?((int32_t)state/2147483648.0f)*2:sinf((float)(j%100003)*.037f+i)*.7f;}
   [buf addObject:x];
  }
  float *expected=malloc(n*4);run(queue,b,a,buf);memcpy(expected,[(id<MTLBuffer>)buf[4] contents],n*4);
  run(queue,p,a,buf);float *got=[(id<MTLBuffer>)buf[4] contents];double delta=0;size_t exact=0;
  for(size_t j=0;j<n;j++){require(isfinite(got[j]));delta=fmax(delta,fabs(got[j]-expected[j]));exact+=got[j]==expected[j];}
  // Independent double oracle for the last query row's first head; this also
  // checks causal length, layer offset, head mapping, and gate application.
  float *q=[(id<MTLBuffer>)buf[0] contents],*k=[(id<MTLBuffer>)buf[1] contents],*v=[(id<MTLBuffer>)buf[2] contents],*g=[(id<MTLBuffer>)buf[3] contents];
  int row=a.rows-1,last=a.pos+row,off=row*a.nh*256,stride=a.nk*256,base=a.layer*a.cap*stride;
  double m=-INFINITY,l=0,acc[256]={0};
  for(int t=0;t<=last;t++){
   double score=0;for(int z=0;z<256;z++)score+=(double)q[off+z]*k[base+t*stride+z];score/=16;
   double next=fmax(m,score),al=exp(m-next),be=exp(score-next);l=l*al+be;
   for(int z=0;z<256;z++)acc[z]=acc[z]*al+be*v[base+t*stride+z];m=next;
  }
  for(int z=0;z<256;z++)require(fabs(got[off+z]-acc[z]/l/(a.gated?1+exp(-(double)g[off+z]):1))<2e-5);
  require(delta==0);double bt=0,pt=0;
  for(int rep=0;rep<6;rep++){if(rep&1){pt+=run(queue,p,a,buf);bt+=run(queue,b,a,buf);}else{bt+=run(queue,b,a,buf);pt+=run(queue,p,a,buf);}}
  printf("%u,%d,%u,%.5f,%.5f,%.3f,%.9g,%zu/%zu\n",a.nh,contexts[c],a.rows,bt/6,pt/6,bt/pt,delta,exact,n);fflush(stdout);free(expected);
 }}
}return 0;}
