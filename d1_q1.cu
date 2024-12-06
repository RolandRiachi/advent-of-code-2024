// Throughout the Chief's office, the historically significant locations are listed not by name but by a unique number called the location ID. 
// To make sure they don't miss anything, The Historians split into two groups, each searching the office and trying to create their own complete list of location IDs.

// There's just one problem: by holding the two lists up side by side (your puzzle input), it quickly becomes clear that the lists aren't very similar. 
// Maybe you can help The Historians reconcile their lists?

// For example: [3,4,2,1,3,3] and [4,3,5,3,9,3]

// Pair up the smallest number in the left list with the smallest number in the right list, then the second-smallest left-number with the second-smallest right number,
// and so on ... Figure out how far apart the numbers are; you'll need to add up all of  these distances.

// TODO:
// [] Read input
// [] Sort a list (bitonic sort?)
// [] Sort both lists
// [] Compute sum of pairwise distances
// [] Print output

#include <stdio.h>

__global__ void VecAbsDiff(const int* A, const int* B, int* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < numElements ) {
        C[i] = abs(A[i] - B[i]);
    }
}

__global__ int SerialParallelSum(const int* A, int numElements) {
    // --- Serial-Parallel Sum Algorithm ---
    // Let p = number of threads, then numElements = (p-1)k + q
    // In parallel, compute S_i = \sum_{j=1}^k A[k*(i-1)+j] for i < p and S_p = \sum_{j=1}^q A[k*(p-1)+j]
    // Afterwards, compute S = \sum_{i=1}^p S_i
    // Finally, return S

    // --- Notes ---
    // Can I reshape the vector A from (N,) to (k, p)

    return 0
}

int main(void) {
    int numElements = 50000;
    size_t size = numElements * sizeof(int);

    // Allocate memory for host vectors
    int *h_A = (int *)malloc( size );
    int *h_B = (int *)malloc( size );
    int *h_C = (int *)malloc( size );

    // Initialize host vectors
    for ( int i = 0; i < numElements; i++  ) {
        h_A[i] = 2*i + 1;
        h_B[i] = i;
    }

    // Declare device vectors
    int *d_A, *d_B, *d_C;

    // Allocate memory for device vectors
    cudaMalloc( (void**)&d_A, size );
    cudaMalloc( (void**)&d_B, size );
    cudaMalloc( (void**)&d_C, size );

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Perform Operation
    int threadsPerBlock = 1024; // Maximum threadsPerBlock in x dim
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock; // Number of blocks necessary given numElements
    VecAbsDiff<<<blocksPerGrid, threadsPerBlock>>>(d_B, d_A, d_C, numElements);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Print result
    // printf("The absolute difference between A and B is ");
    printf("The last value of C is %d\n", h_C[numElements - 1]);
}
