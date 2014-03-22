/*
 * Copyright (c) 2014 NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <math.h>
#include <cuda.h>
#include <builtin_types.h>
#include <drvapi_error_string.h>
#include "nvvm.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <assert.h>

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line );
        exit(-1);
    }
}

CUdevice cudaDeviceInit()
{
    CUdevice cuDevice = 0;
    int deviceCount = 0;
    CUresult err = cuInit(0);
    char name[100];
    int major=0, minor=0;

    if (CUDA_SUCCESS == err)
        checkCudaErrors(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
        exit(-1);
    }
    checkCudaErrors(cuDeviceGet(&cuDevice, 0));
    cuDeviceGetName(name, 100, cuDevice);
    printf("Using CUDA Device [0]: %s\n", name);

    checkCudaErrors( cuDeviceComputeCapability(&major, &minor, cuDevice) );
    printf("compute capability = %d.%d\n", major, minor);
    if (major < 3) {      
        fprintf(stderr, "Device 0 is not sm_30 or later\n");
        exit(-1);
    }
    return cuDevice;
}


CUresult initCUDA(CUcontext *phContext,
                  CUdevice *phDevice,
                  CUmodule *phModule,
                  CUfunction *phKernel,
                  const char *ptx)
{
    CUlinkState linkState;
    void *cubin;
    size_t cubinSize;
    
    // Initialize 
    *phDevice = cudaDeviceInit();

    // Create context on the device
    checkCudaErrors(cuCtxCreate(phContext, 0, *phDevice));

    // Load module from PTX
    checkCudaErrors(cuModuleLoadDataEx(phModule, ptx, 0, NULL, NULL));

    // Locate the kernel entry point
    checkCudaErrors(cuModuleGetFunction(phKernel, *phModule, "test_kernel"));

    return CUDA_SUCCESS;
}

char *loadProgramSource(const char *filename, size_t *size) 
{
    struct stat statbuf;
    FILE *fh;
    char *source = NULL;
    *size = 0;
    fh = fopen(filename, "rb");
    if (fh) {
        stat(filename, &statbuf);
        source = (char *) malloc(statbuf.st_size+1);
        if (source) {
            fread(source, statbuf.st_size, 1, fh);
            source[statbuf.st_size] = 0;
            *size = statbuf.st_size+1;
        }
    }
    else {
        fprintf(stderr, "Error reading file %s\n", filename);
        exit(-1);
    }
    return source;
}

char *generatePTX(const char *ll, size_t size, const char *filename)
{
    nvvmResult result;
    nvvmProgram program;
    size_t PTXSize;
    char *PTX = NULL;
    const char *options[] = { "-arch=compute_30" };

    result = nvvmCreateProgram(&program);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmCreateProgram: Failed\n");
        exit(-1); 
    }

    result = nvvmAddModuleToProgram(program, ll, size, filename);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmAddModuleToProgram: Failed\n");
        exit(-1);
    }
 
  
 
    result = nvvmCompileProgram(program,  1, options);
    if (result != NVVM_SUCCESS) {
        char *Msg = NULL;
        size_t LogSize;
        fprintf(stderr, "nvvmCompileProgram: Failed\n");
        nvvmGetProgramLogSize(program, &LogSize);
        Msg = (char*)malloc(LogSize);
        nvvmGetProgramLog(program, Msg);
        fprintf(stderr, "%s\n", Msg);
        free(Msg);
        exit(-1);
    }
    
    result = nvvmGetCompiledResultSize(program, &PTXSize);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmGetCompiledResultSize: Failed\n");
        exit(-1);
    }
    
    PTX = (char*)malloc(PTXSize);
    result = nvvmGetCompiledResult(program, PTX);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmGetCompiledResult: Failed\n");
        free(PTX);
        exit(-1);
    }
    
    result = nvvmDestroyProgram(&program);
    if (result != NVVM_SUCCESS) {
      fprintf(stderr, "nvvmDestroyProgram: Failed\n");
      free(PTX);
      exit(-1);
    }
    
    return PTX;
}

int main(int argc, char **argv)
{
    const unsigned int nThreads = 1;
    const unsigned int nBlocks  = 1;

    CUcontext    hContext = 0;
    CUdevice     hDevice  = 0;
    CUmodule     hModule  = 0;
    CUfunction   hKernel  = 0;
    char        *ptx      = NULL;
    unsigned int i;
    int          depth    = 0;

    // Pointers to the variables in the managed memory.
    // See uvmlite64.ll for their definition.
    CUdeviceptr devp_xxx, devp_yyy;
    size_t size_xxx, size_yyy;
    int *p_xxx, *p_yyy;
    // Get the ll from file
    size_t size = 0;

#if BUILD_64_BIT
    const char *filename = "uvmlite64.ll";
#else
    #error  uvm-lite only supports 64-bit mode
#endif
    char *ll = loadProgramSource(filename, &size);
    fprintf(stdout, "NVVM IR ll file loaded\n");

    // Use libnvvm to generte PTX
    ptx = generatePTX(ll, size, filename);
    fprintf(stdout, "PTX generated:\n");
    fprintf(stdout, "%s\n", ptx);
    
    // Initialize the device and get a handle to the kernel
    checkCudaErrors(initCUDA(&hContext, &hDevice, &hModule, &hKernel, ptx));

    // Whether or not a device supports unified addressing may be queried by calling
    // cuDeviceGetAttribute() with the deivce attribute CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING.
    {
      int attrVal;
      checkCudaErrors(cuDeviceGetAttribute(&attrVal, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, hDevice));
      assert(attrVal == 1);
    }

    // get the address of the variable xxx, yyy in the managed memory.
    checkCudaErrors(cuModuleGetGlobal(&devp_xxx, &size_xxx, hModule, "xxx"));
    checkCudaErrors(cuModuleGetGlobal(&devp_yyy, &size_yyy, hModule, "yyy"));   

    // Whether or not the pointer points to managed memory may be queried by calling
    // cuPointerGetAttribute() with the pointer attribute CU_POINTER_ATTRIBUTE_IS_MANAGED.
    {
      unsigned int attrVal;

      checkCudaErrors(cuPointerGetAttribute(&attrVal, CU_POINTER_ATTRIBUTE_IS_MANAGED, devp_xxx));
      assert(attrVal == 1);
      checkCudaErrors(cuPointerGetAttribute(&attrVal, CU_POINTER_ATTRIBUTE_IS_MANAGED, devp_yyy));
      assert(attrVal == 1);
    }

    // The "physical" memory location of the memory that the devp_yyy addresses is the device memory type.
    {
      unsigned int attrVal;

      checkCudaErrors(cuPointerGetAttribute(&attrVal, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, devp_xxx));
      assert(attrVal == CU_MEMORYTYPE_DEVICE);
      checkCudaErrors(cuPointerGetAttribute(&attrVal, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, devp_yyy));
      assert(attrVal == CU_MEMORYTYPE_DEVICE);
    }

    // Since CUdeviceptr is opaque, it is safe to use cuPointerGetAttribute to get the host pointers.
    {
      void *host_ptr_xxx, *host_ptr_yyy;
      
      checkCudaErrors(cuPointerGetAttribute(&host_ptr_xxx, CU_POINTER_ATTRIBUTE_HOST_POINTER, devp_xxx));
      checkCudaErrors(cuPointerGetAttribute(&host_ptr_yyy, CU_POINTER_ATTRIBUTE_HOST_POINTER, devp_yyy));
      
      p_xxx = (int *)host_ptr_xxx;
      p_yyy = (int *)host_ptr_yyy;
    }

    printf("The initial value of xxx initialized by the device = %d\n", *p_xxx);
    printf("The initial value of yyy initialized by the device = %d\n", *p_yyy);

    assert(*p_xxx == 10);
    assert(*p_yyy == 100);

    // host adds 1 and 11 to xxx and yyy
    *p_xxx += 1;
    *p_yyy += 11;

    printf("The host added 1 and 11 to xxx and yyy.\n");


    // Launch the kernel
    // Kernel parameters
    {
      void *params[] = { (void *)&devp_xxx };

      checkCudaErrors(cuLaunchKernel(hKernel, nBlocks, 1, 1, nThreads, 1, 1, 0, NULL, params, NULL));
    }

    checkCudaErrors(cuCtxSynchronize());

    printf("kernel added 20 and 30 to xxx and yyy, respectively.\n");
    printf("The final value checked in the host: xxx = %d, yyy = %d\n", *p_xxx, *p_yyy);

    if (hModule) {
        checkCudaErrors(cuModuleUnload(hModule));
        hModule = 0;
    }
    if (hContext) {
        checkCudaErrors(cuCtxDestroy(hContext));
        hContext = 0;
    }

    free(ll);
    free(ptx);
    
    return 0;
}
