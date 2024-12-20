#include <stdio.h>

__global__ void g_vecDiff(const int *A, const int *B, int *C, int N) {
    int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if ( thread_idx < N ) {
        C[thread_idx] = A[thread_idx] - B[thread_idx];
    }
}

void vecDiff(const int *A, const int *B, int *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] - B[i];
    }
}

__host__ void h_printfVec(const int *vec, int N) {
    printf("[ ");
    for (int i = 0; i < N; i++) {
        printf("%d ", vec[i]);
    }
    printf("]\n");
}

int main(void) {
    int numElements = 1 << 4;
    size_t size = numElements * sizeof(int);

    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    h_A = (int *)malloc(size);
    h_B = (int *)malloc(size);
    h_C = (int *)malloc(size);

    cudaError_t err;
    err = cudaMalloc(&d_A, size);
    err = cudaMalloc(&d_B, size);
    err = cudaMalloc(&d_C, size);

    for (int i = 0; i < numElements; i++){
        h_A[i] = i;
        h_B[i] = 1;
    }

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements / threadsPerBlock);
    if ( blocksPerGrid == 0 ) blocksPerGrid = 1;
    printf("%d\n", 0 | 1);
    g_vecDiff<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    vecDiff(h_A, h_B, h_C, numElements);
    //h_printfVec(h_C, numElements);

    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("The result is ");
    // h_printfVec(h_C, numElements);
}
