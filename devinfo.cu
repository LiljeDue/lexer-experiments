#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int d = 0; d < nDevices; d++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);
        int shmem_optin = 0;
        cudaDeviceGetAttribute(&shmem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, d);
        printf("Device %d: %s\n", d, prop.name);
        printf("  sharedMemPerBlock:      %zu bytes (%zu KB)\n",
               prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024);
        printf("  sharedMemPerBlockOptin: %d bytes (%d KB)\n",
               shmem_optin, shmem_optin / 1024);
    }
    return 0;
}
