#include <cuda_runtime.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum {
    PROBE_BYTES = 256 * 1024 * 1024,
    PROBE_THREADS = 256,
    PROBE_BLOCKS = 512,
};

static void fail_cuda(cudaError_t error, const char *operation) {
    if (error == cudaSuccess) {
        return;
    }
    fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    exit(1);
}

__global__ static void read_once(const uint4 *input, size_t count,
                                 uint32_t *partial) {
    const size_t thread = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint32_t sum = 0;
    for (size_t index = thread; index < count; index += stride) {
        const uint4 value = input[index];
        sum += value.x ^ value.y ^ value.z ^ value.w;
    }
    partial[thread] = sum;
}

static double time_read(const void *device_pointer, uint32_t *partial,
                        uint32_t *checksum) {
    cudaEvent_t start;
    cudaEvent_t stop;
    fail_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    fail_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    fail_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    read_once<<<PROBE_BLOCKS, PROBE_THREADS>>>(
        (const uint4 *)device_pointer, PROBE_BYTES / sizeof(uint4), partial);
    fail_cuda(cudaGetLastError(), "read_once launch");
    fail_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    fail_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float milliseconds = 0.0f;
    fail_cuda(cudaEventElapsedTime(&milliseconds, start, stop),
              "cudaEventElapsedTime");

    uint32_t *host_partial = (uint32_t *)malloc(
        (size_t)PROBE_BLOCKS * PROBE_THREADS * sizeof(*host_partial));
    if (!host_partial) {
        fprintf(stderr, "malloc partial results failed\n");
        exit(1);
    }
    fail_cuda(cudaMemcpy(host_partial, partial,
                         (size_t)PROBE_BLOCKS * PROBE_THREADS *
                             sizeof(*host_partial),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(partial)");
    uint32_t sum = 0;
    for (size_t index = 0; index < (size_t)PROBE_BLOCKS * PROBE_THREADS;
         ++index) {
        sum ^= host_partial[index];
    }
    free(host_partial);
    fail_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    fail_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    *checksum = sum;
    return milliseconds;
}

static double gb_per_second(double milliseconds) {
    return (double)PROBE_BYTES / 1.0e9 / (milliseconds / 1.0e3);
}

int main(void) {
    int device = 0;
    cudaDeviceProp properties;
    fail_cuda(cudaGetDeviceProperties(&properties, device),
              "cudaGetDeviceProperties");
    fail_cuda(cudaSetDevice(device), "cudaSetDevice");

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    fail_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    uint32_t *partial = NULL;
    fail_cuda(cudaMalloc(&partial,
                         (size_t)PROBE_BLOCKS * PROBE_THREADS *
                             sizeof(*partial)),
              "cudaMalloc(partial)");

    void *device_memory = NULL;
    fail_cuda(cudaMalloc(&device_memory, PROBE_BYTES),
              "cudaMalloc(device probe)");
    fail_cuda(cudaMemset(device_memory, 0xa5, PROBE_BYTES),
              "cudaMemset(device probe)");
    uint32_t device_checksum = 0;
    const double device_ms = time_read(device_memory, partial,
                                       &device_checksum);

    void *managed_memory = NULL;
    fail_cuda(cudaMallocManaged(&managed_memory, PROBE_BYTES),
              "cudaMallocManaged");
    memset(managed_memory, 0xa5, PROBE_BYTES);
    fail_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(managed)");
    uint32_t managed_checksum = 0;
    const double managed_ms = time_read(managed_memory, partial,
                                        &managed_checksum);

    void *mapped_memory = mmap(NULL, PROBE_BYTES, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped_memory == MAP_FAILED) {
        fprintf(stderr, "mmap: %s\n", strerror(errno));
        return 1;
    }
    memset(mapped_memory, 0xa5, PROBE_BYTES);
    fail_cuda(cudaHostRegister(mapped_memory, PROBE_BYTES,
                               cudaHostRegisterReadOnly),
              "cudaHostRegister");
    void *mapped_device_pointer = NULL;
    fail_cuda(cudaHostGetDevicePointer(&mapped_device_pointer, mapped_memory,
                                       0),
              "cudaHostGetDevicePointer");
    uint32_t mapped_checksum = 0;
    const double mapped_ms = time_read(mapped_device_pointer, partial,
                                       &mapped_checksum);

    printf("{\n");
    printf("  \"device\": \"%s\",\n", properties.name);
    printf("  \"compute_capability\": \"%d.%d\",\n", properties.major,
           properties.minor);
    printf("  \"integrated\": %d,\n", properties.integrated);
    printf("  \"can_map_host_memory\": %d,\n",
           properties.canMapHostMemory);
    printf("  \"managed_memory\": %d,\n", properties.managedMemory);
    printf("  \"concurrent_managed_access\": %d,\n",
           properties.concurrentManagedAccess);
    printf("  \"pageable_memory_access\": %d,\n",
           properties.pageableMemoryAccess);
    printf("  \"total_memory_bytes\": %zu,\n", total_bytes);
    printf("  \"free_memory_bytes\": %zu,\n", free_bytes);
    printf("  \"probe_bytes\": %d,\n", PROBE_BYTES);
    printf("  \"device_read_ms\": %.3f,\n", device_ms);
    printf("  \"device_read_gbps\": %.3f,\n", gb_per_second(device_ms));
    printf("  \"managed_read_ms\": %.3f,\n", managed_ms);
    printf("  \"managed_read_gbps\": %.3f,\n", gb_per_second(managed_ms));
    printf("  \"registered_mmap_read_ms\": %.3f,\n", mapped_ms);
    printf("  \"registered_mmap_read_gbps\": %.3f,\n",
           gb_per_second(mapped_ms));
    printf("  \"checksums_match\": %s\n",
           device_checksum == managed_checksum &&
                   device_checksum == mapped_checksum
               ? "true"
               : "false");
    printf("}\n");

    fail_cuda(cudaHostUnregister(mapped_memory), "cudaHostUnregister");
    if (munmap(mapped_memory, PROBE_BYTES) != 0) {
        fprintf(stderr, "munmap: %s\n", strerror(errno));
        return 1;
    }
    fail_cuda(cudaFree(managed_memory), "cudaFree(managed)");
    fail_cuda(cudaFree(device_memory), "cudaFree(device)");
    fail_cuda(cudaFree(partial), "cudaFree(partial)");
    return 0;
}
