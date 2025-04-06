#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <malloc.h>


/*
 extern "C"{__global__ void gStub(float* a, float* b){
//    int tid=threadIdx.x+blockIdx.x*blockDim.x;
//    a[tid]=a[tid]+b[tid];
}
 }
*/
int main(){
    int N=2048;
    float* a=(float*)calloc(N, sizeof(float));
    float* b=(float*)calloc(N, sizeof(float));

 cuInit(0);
     //Pobranie handlera do devica
    //(moze byc kilka urzadzen. Tutaj zakladamy, ze jest conajmniej jedno)
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot acquire device 0\n");
        exit(1);
    }

    //Tworzy kontext
    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        printf("cannot create context\n");
        exit(1);
    }


    for(int i=0; i<N; i++){
        a[i]=2*i;
        b[i]=2*i+1;
    }

    float *a_d, *b_d;
    cudaMalloc((void**)&a_d, N*sizeof(float));
    cudaMalloc((void**)&b_d, N*sizeof(float));

    cudaMemcpy(a_d, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, N*sizeof(float), cudaMemcpyHostToDevice);

    //gStub<<<N/128,128>>>(a_d,b_d);
    //cudaDeviceSynchronize();
 CUmodule cuModule = (CUmodule)0;
 cuModuleLoad(&cuModule, "cda.ptx");
 CUfunction gStub;
 cuModuleGetFunction(&gStub, cuModule, "gStub");
 printf("%X\n", gStub);
 void* args[] = {&a_d, &b_d};
//    size_t argBufferSize;
//           char argBuffer[256];
//
//           // populate argBuffer and argBufferSize
//
//           void *config[] = {
//               CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
//               CU_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,
//               CU_LAUNCH_PARAM_END
//           };
//           //status = cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sh, s, NULL, config);

 //printf("%X\t%X\n", args[0],args[1]);
 cuLaunchKernel(gStub, N/128, 1, 1, 128, 1, 1, 0, 0, args, 0);

    cudaMemcpy(a, a_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i<N; i+=N/16)
        printf("%g\n",a[i]);

    cuCtxDestroy(cuContext);
    return 0;
}
