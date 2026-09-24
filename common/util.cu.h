#include <cstdint>
#include <assert.h>
#include <cuda_runtime.h>
#define gpuAssert(x) _gpuAssert(x, __FILE__, __LINE__)

// Prints mean time (μs), a 95% confidence interval of the mean, and mean
// throughput in GB/s for `bytes` moved per run. measurements are in ms.
void compute_descriptors(float* measurements, size_t size, size_t bytes) {
    double sample_mean = 0;
    double sample_sq_mean = 0;
    double sample_gbps = 0;
    double factor = (double)bytes / (1000.0 * size);   // bytes/μs / 1000 = GB/s

    for (size_t i = 0; i < size; i++) {
        double diff = max(1e3 * measurements[i], 0.5);   // μs
        sample_mean += diff / size;
        sample_sq_mean += (diff * diff) / size;
        sample_gbps += factor / diff;
    }
    // Sample standard deviation (Bessel-corrected), 95% CI = 1.96 * std / sqrt(n).
    double sample_variance = (sample_sq_mean - sample_mean * sample_mean) * size / max(size - 1, (size_t)1);
    double sample_std = sqrt(max(sample_variance, 0.0));
    double bound = (1.96 * sample_std) / sqrt((double)size);

    printf("%.0lfμs ", sample_mean);
    printf("(95%% CI: [%.1lfμs, %.1lfμs]); ", sample_mean - bound, sample_mean + bound);
    printf("%.0lfGB/s\n", sample_gbps);
}

void _gpuAssert(cudaError_t code, const char *fname, int lineno) {
    if(code != cudaSuccess) {
        printf("GPU Error: %s, File: %s, Line: %i\n", cudaGetErrorString(code), fname, lineno);
        fflush(stdout);
        exit(1);
    }
}
