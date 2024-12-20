#include <stdio.h>

#define BYTE_SIZE 256
#define NUM_BYTES 2

__host__ void h_printfVec(int *vector, int N) {
    printf("[ ");
    for (int i = 0; i < N; i++) {
        printf("%d ", vector[i]);
    }
    printf("]\n");
}

__host__ void h_printfVec(unsigned char *vector, int N) {
    printf("[ ");
    for (int i = 0; i < N; i++) {
        printf("%d ", (int)vector[i]);
    }
    printf("]\n");
}

__host__ void h_naiveByteRadixSort(unsigned char *vector, int *sortedBuffer, int N) {

    for (int i=0; i<N; i++) {
        unsigned char c = vector[i];
        sortedBuffer[c] = c;
    }

}

__host__ void h_byteRadixSort(int *vector, int *destinationBuffer, int N, int pass) {
    int counters[BYTE_SIZE];
    memset(counters, 0, BYTE_SIZE*sizeof(int));

    for ( int i=0; i<N; i++ ) {
        unsigned char c = vector[i];
        unsigned char radix = (vector[i]>>(pass<<3)) & (NUM_BYTES * BYTE_SIZE);
        printf("%x\n", vector[i]);
        counters[c]++;
    }

    h_printfVec(counters, BYTE_SIZE);

    int offsetTable[BYTE_SIZE];
    offsetTable[0] = 0;
    for ( int i=1; i<BYTE_SIZE; i++ ) {
        offsetTable[i] = offsetTable[i-1] + counters[i-1];
    }

    for ( int i=0; i<N; i++ ) {
        int c = (int)vector[i];
        destinationBuffer[offsetTable[c]++] = c;
    }
}

__host__ void h_radixSort(float *vector, int N) {
}

__global__ void g_vecRadixSort(float *vector, int N) {
}

int main(void) {
    int numElements = 1 << 4;
    size_t size = numElements * sizeof(float);

    int *h_vector, *d_vector;

    h_vector = (int *)malloc(size);

    for (int i = 0; i < numElements; i++) {
        // h_vector[i] = (float)rand() / (float)RAND_MAX;
        h_vector[i] = rand() / ( RAND_MAX / (NUM_BYTES * BYTE_SIZE) );
    }
    
    h_printfVec( h_vector, numElements);

    // int h_sortedBuffer[BYTE_SIZE];
    // memset(h_sortedBuffer, -1, BYTE_SIZE*sizeof(int));

    // h_naiveByteRadixSort(h_vector, h_sortedBuffer, numElements);
    // h_printfVec( h_sortedBuffer, BYTE_SIZE);


    int *h_destinationBuffer;
    h_destinationBuffer = (int *)malloc( numElements * sizeof(int) );

    h_byteRadixSort(h_vector, h_destinationBuffer, numElements, 0);
    h_printfVec( h_destinationBuffer, numElements );

    
}
