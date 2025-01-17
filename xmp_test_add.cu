#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/cpu_support.h"
#include "../utility/cpu_simple_bn_math.h"
#include "../utility/gpu_support.h"
#include <vector>
#include <chrono>

using namespace std;

/************************************************************************************************
 *  This example performs component-wise addition of two arrays of 1024-bit bignums.
 *
 *  The example uses a number of utility functions and macros:
 *
 *    random_words(uint32_t *words, uint32_t count)
 *       fills words[0 .. count-1] with random data
 *
 *    add_words(uint32_t *r, uint32_t *a, uint32_t *b, uint32_t count) 
 *       sets bignums r = a+b, where r, a, and b are count words in length
 *
 *    compare_words(uint32_t *a, uint32_t *b, uint32_t count)
 *       compare bignums a and b, where a and b are count words in length.
 *       return 1 if a>b, 0 if a==b, and -1 if b>a
 *    
 *    CUDA_CHECK(call) is a macro that checks a CUDA result for an error,
 *    if an error is present, it prints out the error, call, file and line.
 *
 *    CGBN_CHECK(report) is a macro that checks if a CGBN error has occurred.
 *    if so, it prints out the error, and instance information
 *
 ************************************************************************************************/
 
// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 4
#define BITS 1024
// #define INSTANCES 100000

// Declare the instance type
typedef struct {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> sum;
} instance_t;



// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

// the actual kernel
__global__ void kernel_add(cgbn_error_report_t *report, cgbn_mem_t<BITS> *ds, cgbn_mem_t<BITS> *da,cgbn_mem_t<BITS> *db,uint32_t count) {
  int32_t instance;
  
  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  context_t      bn_context(cgbn_report_monitor, report, instance);   // construct a context
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  a, b, r;                                             // define a, b, r as 1024-bit bignums

  cgbn_load(bn_env, a, &(da[instance]));      // load my instance's a value
  cgbn_load(bn_env, b, &(db[instance]));      // load my instance's b value
  cgbn_add(bn_env, r, a, b);                           // r=a+b
  cgbn_store(bn_env, &(ds[instance]), r);   // store r into sum
}

int main(int argc, char *argv[]) {
    int INSTANCES = atoi(argv[1]);
  cgbn_error_report_t *report;
  
  printf("Genereating instances ...\n");
  cgbn_mem_t<BITS> *ha, *hb, *hs;
  cgbn_mem_t<BITS> *da, *db, *ds;
  ha = (cgbn_mem_t<BITS> *)(malloc(INSTANCES * sizeof(cgbn_mem_t<BITS>)));
  hb = (cgbn_mem_t<BITS> *)(malloc(INSTANCES * sizeof(cgbn_mem_t<BITS>)));
  hs = (cgbn_mem_t<BITS> *)(malloc(INSTANCES * sizeof(cgbn_mem_t<BITS>)));
  for(int index=0;index<INSTANCES;index++) {
    random_words(ha[index]._limbs, BITS/32);
    random_words(hb[index]._limbs, BITS/32);
  }
  

  printf("Copying instances to the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&da, sizeof(cgbn_mem_t<BITS>)*INSTANCES));
  CUDA_CHECK(cudaMalloc((void **)&db, sizeof(cgbn_mem_t<BITS>)*INSTANCES));
  CUDA_CHECK(cudaMalloc((void **)&ds, sizeof(cgbn_mem_t<BITS>)*INSTANCES));
  CUDA_CHECK(cudaMemcpy(da, ha, sizeof(cgbn_mem_t<BITS>)*INSTANCES, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb, sizeof(cgbn_mem_t<BITS>)*INSTANCES, cudaMemcpyHostToDevice));
  
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");
  // launch with TPI threads per instance, 256 threads (256/TPI instances) per block
  // warm_up
    int blockThreadNum = 256;
    int blockInstanceNum = blockThreadNum / TPI; // 256 / 4 = 64
    for (int i = 0; i < 5; i++) {
        kernel_add<<<(INSTANCES+ blockInstanceNum - 1)/blockInstanceNum, 256>>>(report, ds, da, db, INSTANCES);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    int reapte = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reapte; i++) {
        kernel_add<<<(INSTANCES + blockInstanceNum - 1)/blockInstanceNum, 256>>>(report, ds, da, db, INSTANCES);

    } 
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double t =  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /1000.0/reapte;
    double ops = 1000.0 * INSTANCES / t;
    printf("time: %f ms\n", t);
    printf("ops: %f os/s\n", ops);
    printf("bandwidth: %f GB/s\n", 1000.0*3*BITS*INSTANCES/8/t/1024/1024/1024);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
    
  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(hs, ds, sizeof(cgbn_mem_t<BITS>)*INSTANCES, cudaMemcpyDeviceToHost));
  
  free(ha);
  free(hb);
  free(hs);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(ds));
  CUDA_CHECK(cgbn_error_report_free(report));
}

