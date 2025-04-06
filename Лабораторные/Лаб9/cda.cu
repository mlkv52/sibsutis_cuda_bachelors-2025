extern "C"
__global__ void gEasy1(float* a, float* b){
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    a[i]=a[i]+b[i];
}
