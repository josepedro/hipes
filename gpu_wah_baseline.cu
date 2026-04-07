#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h> 
#include <thrust/extrema.h>     
#include <cuda_runtime.h>

#define BM31 0x7FFFFFFF
#define BMMSB 0x80000000

uint32_t* algorithm3_decompress_bitmap(const uint32_t* C, uint32_t m, uint32_t* n);
int* bitmap_to_set(const uint32_t* decompressed_bitmap, uint32_t num_words, uint32_t* set_size);
int compare_ints(const void* a, const void* b);

__global__ void set_to_bitmap_kernel(const int* d_set, uint32_t* d_bitmap, uint32_t set_size) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < set_size) {
        int value = d_set[i];

        uint32_t word_idx = value / 32;
        uint32_t bit_pos = value % 32;
        uint32_t mask = (1U << bit_pos);

        atomicOr(&d_bitmap[word_idx], mask);
    }
}

__global__ void extend_kernel(const uint32_t* B, uint32_t num_bits, uint32_t* E) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_subbitmaps = (num_bits + 30) / 31;
    if (num_bits > 28 && num_bits % 31 == 0) {
        num_subbitmaps = num_bits / 31;
    }

    if (i >= num_subbitmaps) return;

    uint64_t start_bit = (uint64_t)i * 31;
    uint32_t start_word = start_bit / 32;
    uint32_t bit_offset = start_bit % 32;
    uint32_t num_words_raw = (num_bits + 31) / 32;

    if (start_word >= num_words_raw) return;

    uint64_t temp_word = (uint64_t)(B[start_word]) >> bit_offset;
    if (bit_offset != 0 && (start_word + 1) < num_words_raw) {
        temp_word |= (uint64_t)B[start_word + 1] << (32 - bit_offset);
    }

    E[i] = (uint32_t)(temp_word & BM31);
}

__global__ void mark_prefill_kernel(uint32_t* E, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t word = E[i];
    if (word == 0x0 || word == BM31) {
        E[i] = word | BMMSB; 
    }
}

__global__ void find_boundaries_kernel(const uint32_t* E, uint32_t* F, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    if ((E[i] & BMMSB) == 0 || (E[i] & 0xC0000000) != (E[i + 1] & 0xC0000000)) {
        F[i] = 1;
    } else {
        F[i] = 0;
    }
}

__global__ void create_T_array_kernel(const uint32_t* F, const uint32_t* SF, uint32_t* T, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (F[i] == 1) {
        T[SF[i]] = i + 1;
    }
}

__global__ void compress_kernel(const uint32_t* E, const uint32_t* T, uint32_t* C, uint32_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    uint32_t j = T[i] - 1; 
    uint32_t X = E[j];     

    if ((X & BMMSB) != 0) { 
        uint32_t count = (i == 0) ? T[0] : T[i] - T[i-1];

        X = (X & 0xC0000000) | (count & 0x3FFFFFFF);
    }
    C[i] = X;
}

uint32_t get_max_from_set_thrust(const int* h_set, uint32_t set_size) {

    thrust::device_vector<int> d_set(h_set, h_set + set_size);

    int max_val = *thrust::max_element(d_set.begin(), d_set.end());

    return static_cast<uint32_t>(max_val);
}

uint32_t* algorithm1_extend_bitmap_gpu(const uint32_t* B, uint32_t num_bits, uint32_t* n) {
    uint32_t num_words_raw = (num_bits + 31) / 32;
    uint32_t num_subbitmaps = (num_bits + 30) / 31;
    if (num_bits > 28 && num_bits % 31 == 0) {
        num_subbitmaps = num_bits / 31;
    }
    *n = num_subbitmaps;

    uint32_t* d_B;
    uint32_t* d_E;
    cudaMalloc((void**)&d_B, num_words_raw * sizeof(uint32_t));
    cudaMalloc((void**)&d_E, *n * sizeof(uint32_t));

    cudaMemcpy(d_B, B, num_words_raw * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (*n + threadsPerBlock - 1) / threadsPerBlock;
    extend_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_B, num_bits, d_E);

    uint32_t* h_E = (uint32_t*)malloc(*n * sizeof(uint32_t));
    cudaMemcpy(h_E, d_E, *n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_B);
    cudaFree(d_E);

    return h_E;
}

uint32_t* algorithm2_compress_bitmap_gpu(const uint32_t* E_orig, uint32_t n, uint32_t* m) {
    uint32_t *d_E, *d_F, *d_SF, *d_T, *d_C;

    cudaMalloc((void**)&d_E, n * sizeof(uint32_t));
    cudaMalloc((void**)&d_F, n * sizeof(uint32_t));
    cudaMalloc((void**)&d_SF, n * sizeof(uint32_t));

    cudaMemcpy(d_E, E_orig, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    mark_prefill_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_E, n);

    find_boundaries_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_E, d_F, n);
    uint32_t one = 1;
    cudaMemcpy(d_F + (n - 1), &one, sizeof(uint32_t), cudaMemcpyHostToDevice);

    thrust::device_ptr<uint32_t> d_F_ptr(d_F);
    thrust::device_ptr<uint32_t> d_SF_ptr(d_SF);
    thrust::exclusive_scan(d_F_ptr, d_F_ptr + n, d_SF_ptr);

    uint32_t last_SF, last_F;
    cudaMemcpy(&last_SF, d_SF + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_F,  d_F + (n - 1),  sizeof(uint32_t), cudaMemcpyDeviceToHost);
    *m = last_SF + last_F;

    cudaMalloc((void**)&d_T, *m * sizeof(uint32_t));
    create_T_array_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_F, d_SF, d_T, n);

    cudaMalloc((void**)&d_C, *m * sizeof(uint32_t));
    int compressBlocks = (*m + threadsPerBlock - 1) / threadsPerBlock;
    compress_kernel<<<compressBlocks, threadsPerBlock>>>(d_E, d_T, d_C, *m);

    uint32_t* h_C = (uint32_t*)malloc(*m * sizeof(uint32_t));
    cudaMemcpy(h_C, d_C, *m * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_E);
    cudaFree(d_F);
    cudaFree(d_SF);
    cudaFree(d_T);
    cudaFree(d_C);

    return h_C;
}

int main(void) {

    const char* filename = "../dataset_8192_MB.txt";
    printf("--- Reading numbers from file: %s ---\n", filename);

    FILE* inputFile = fopen(filename, "r");
    if (inputFile == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return 1;
    }

    int* numbers = NULL;
    size_t count = 0;
    size_t capacity = 0;
    int number;

    while (fscanf(inputFile, "%d", &number) == 1) {
        if (count == capacity) {

            capacity = (capacity == 0) ? 1024 : capacity * 2;
            int* new_numbers = (int*)realloc(numbers, capacity * sizeof(int));
            if (new_numbers == NULL) {
                fprintf(stderr, "Error: Memory reallocation failed.\n");
                free(numbers);
                fclose(inputFile);
                return 1;
            }
            numbers = new_numbers;
        }
        numbers[count++] = number;
    }
    fclose(inputFile);

    printf("Successfully read %zu numbers.\n", count);

    qsort(numbers, count, sizeof(int), compare_ints);

    int* setA = numbers;
    uint32_t set_size = count;

    float repetitions = 100.0;
    int size_measurements = (int) repetitions;
    float measurements_total[size_measurements];
    float measurements_total_compression_speed[size_measurements];
    for (int i = 0; i < size_measurements; i++) {
        measurements_total[i] = 0.0;
        measurements_total_compression_speed[i] = 0.0;
    }
    uint32_t extended_size;
    uint32_t compressed_size;
    uint32_t* extended_bitmap;
    uint32_t* compressed_bitmap;
    uint32_t* raw_bitmap;
    for (int i = 0; i < repetitions + 2; i++) {

        cudaEvent_t start_cuda, stop_cuda;
        cudaEventCreate(&start_cuda);
        cudaEventCreate(&stop_cuda);
        float milliseconds = 0;
        cudaEventRecord(start_cuda);

        uint32_t max_val = get_max_from_set_thrust(setA, set_size);
        uint32_t bitmap_bits = max_val + 1;
        uint32_t bitmap_words = (bitmap_bits + 31) / 32;

        int* d_set;
        uint32_t* d_bitmap;
        cudaMalloc((void**)&d_set, set_size * sizeof(int));
        cudaMalloc((void**)&d_bitmap, bitmap_words * sizeof(uint32_t));

        cudaMemset(d_bitmap, 0, bitmap_words * sizeof(uint32_t));

        cudaMemcpy(d_set, setA, set_size * sizeof(int), cudaMemcpyHostToDevice);

        int threadsPerBlock_set = 256;
        int blocksPerGrid_set = (set_size + threadsPerBlock_set - 1) / threadsPerBlock_set;
        set_to_bitmap_kernel<<<blocksPerGrid_set, threadsPerBlock_set>>>(d_set, d_bitmap, set_size);

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error after set_to_bitmap_kernel: %s\n", cudaGetErrorString(err));
        }

        raw_bitmap = (uint32_t*)malloc(bitmap_words * sizeof(uint32_t));
        cudaMemcpy(raw_bitmap, d_bitmap, bitmap_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        cudaFree(d_set);
        cudaFree(d_bitmap);

        extended_bitmap = algorithm1_extend_bitmap_gpu(raw_bitmap, bitmap_bits, &extended_size);

        compressed_bitmap = algorithm2_compress_bitmap_gpu(extended_bitmap, extended_size, &compressed_size);

        cudaEventRecord(stop_cuda);
        cudaEventSynchronize(stop_cuda);
        cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda);
        printf("Time for WAH Bitmap Compression: %.3f ms\n", milliseconds);

        size_t initial_size_bytes = set_size * sizeof(int);
        uint32_t compressed_size_bytes = compressed_size * sizeof(uint32_t);
        double compression_ratio = (double)initial_size_bytes / compressed_size_bytes;
        printf("Compression Ratio (Uncompressed/Compressed): %.2f to 1\n", compression_ratio);

        double bits_per_byte = (double)compressed_size_bytes * 8.0 / initial_size_bytes;
        printf("Effectiveness (Bits Per Byte): %.3f bpb\n", bits_per_byte);

        double initial_size_MB = (double)initial_size_bytes / (1024.0 * 1024.0);
        double time_s = milliseconds / 1000.0;
        double compression_speed_MBps = initial_size_MB / time_s;
        printf("Compression Speed: %.2f MB/s\n", compression_speed_MBps);
        break;
        if (i > 1) { 
            measurements_total[i - 2] = milliseconds / 1000.0;
            size_t initial_size_bytes = set_size * sizeof(int);
            double initial_size_MB = (double)initial_size_bytes / (1024.0 * 1024.0);
            measurements_total_compression_speed[i - 2] = initial_size_MB / (milliseconds / 1000.0);
        }

        free(raw_bitmap);
        free(extended_bitmap);
        free(compressed_bitmap);
    }
    return 0;
    float milliseconds = measurements_total[0] * 1000;
    uint32_t compressed_size_bytes = compressed_size * sizeof(uint32_t);

    printf("\n--- Compression Statistics ---\n");
    size_t initial_size_bytes = set_size * sizeof(int);
    printf("Initial set size: %zu bytes\n", initial_size_bytes);
    printf("Compressed bitmap size: %u bytes\n", compressed_size_bytes);
    printf("Total GPU compression time: %.3f ms\n", milliseconds);

    if (initial_size_bytes > 0 && compressed_size_bytes > 0 && milliseconds > 0) {

        double compression_ratio = (double)initial_size_bytes / compressed_size_bytes;
        printf("Compression Ratio (Uncompressed/Compressed): %.2f to 1\n", compression_ratio);

        double bits_per_byte = (double)compressed_size_bytes * 8.0 / initial_size_bytes;
        printf("Effectiveness (Bits Per Byte): %.3f bpb\n", bits_per_byte);

        double initial_size_MB = (double)initial_size_bytes / (1024.0 * 1024.0);
        double time_s = milliseconds / 1000.0;
        double compression_speed_MBps = initial_size_MB / time_s;
        printf("Compression Speed: %.2f MB/s\n", compression_speed_MBps);

        printf("\n__________________________________________________________________________________________________\n");

        float total_sum = 0;
        float total_max = 0;
        float total_min = measurements_total[0];
        float total_min_compression_speed = measurements_total_compression_speed[0];
        float total_max_compression_speed = 0;
        float total_sum_compression_speed = 0;
        for (int i = 0; i < repetitions; i++) {
            total_sum += measurements_total[i];
            total_sum_compression_speed += measurements_total_compression_speed[i];
            if (measurements_total_compression_speed[i] > total_max_compression_speed) {
                total_max_compression_speed = measurements_total_compression_speed[i];
            }
            if (measurements_total_compression_speed[i] < total_min_compression_speed) {
                total_min_compression_speed = measurements_total_compression_speed[i];
            }

            if (measurements_total[i] > total_max) {
                total_max = measurements_total[i];
            }
            if (measurements_total[i] < total_min) {
                total_min = measurements_total[i];
            }
        }
        float total_avg = total_sum / repetitions;
        float total_avg_compression_speed = total_sum_compression_speed / repetitions;

        float total_stddev = 0;
        float total_stddev_compression_speed = 0;
        for (int i = 0; i < repetitions; i++) {
            total_stddev += (measurements_total[i] - total_avg) * (measurements_total[i] - total_avg);
            total_stddev_compression_speed += (measurements_total_compression_speed[i] - total_avg_compression_speed) * (measurements_total_compression_speed[i] - total_avg_compression_speed);
        }
        total_stddev = sqrt(total_stddev / repetitions);
        total_stddev_compression_speed = sqrt(total_stddev_compression_speed / repetitions);

        printf("========================================\n");
        printf("Statistics (%.0f repetitions):\n", repetitions);
        printf("========================================\n");
        printf("Total time (avg): %.6f seconds\n", total_avg);
        printf("Total time (max): %.6f seconds\n", total_max);
        printf("Total time (min): %.6f seconds\n", total_min);
        printf("Total time (stddev): %.6f seconds\n", total_stddev);

        printf("Compression Speed (avg): %.6f MB/s\n", total_avg_compression_speed);
        printf("Compression Speed (max): %.6f MB/s\n", total_max_compression_speed);
        printf("Compression Speed (min): %.6f MB/s\n", total_min_compression_speed);
        printf("Compression Speed (stddev): %.6f MB/s\n", total_stddev_compression_speed);
        printf("========================================\n");

        printf("GPU-WAH experiment finished.\n");
        printf("\n__________________________________________________________________________________________________\n");

    }

}

int compare_ints(const void* a, const void* b) {
    int arg1 = *(const int*)a;
    int arg2 = *(const int*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

uint32_t* algorithm3_decompress_bitmap(const uint32_t* C, uint32_t m, uint32_t* n) {

    uint32_t* S = (uint32_t*)malloc(m * sizeof(uint32_t));
    for (uint32_t i = 0; i < m; ++i) {
        if ((C[i] & BMMSB) == 0) { 
            S[i] = 1;
        } else { 
            S[i] = C[i] & 0x3FFFFFFF;
        }
    }

    uint32_t* SS = (uint32_t*)malloc(m * sizeof(uint32_t));
    SS[0] = 0;
    for (uint32_t i = 1; i < m; ++i) {
        SS[i] = SS[i - 1] + S[i - 1];
    }
    *n = SS[m - 1] + S[m - 1];

    uint32_t* SF = (uint32_t*)malloc(*n * sizeof(uint32_t));
    uint32_t current_c_idx = 0;
    for (uint32_t i = 0; i < *n; ++i) {
        if ((current_c_idx + 1 < m) && (i >= SS[current_c_idx + 1])) {
            current_c_idx++;
        }
        SF[i] = current_c_idx;
    }

    uint32_t* E = (uint32_t*)malloc(*n * sizeof(uint32_t));
    for (uint32_t i = 0; i < *n; ++i) {
        uint32_t D = C[SF[i]];
        if ((D & BMMSB) == 0) { 
            E[i] = D;
        } else { 
            if ((D & 0x40000000) == 0) { 
                E[i] = 0;
            } else { 
                E[i] = BM31;
            }
        }
    }

    free(S);
    free(SS);
    free(SF);
    return E;
}

int* bitmap_to_set(const uint32_t* decompressed_bitmap, uint32_t num_words, uint32_t* set_size) {
    int* temp_set = (int*)malloc(num_words * 31 * sizeof(int));
    uint32_t count = 0;

    for (uint32_t i = 0; i < num_words; i++) {
        uint32_t word = decompressed_bitmap[i];
        if (word == 0) continue;

        for (int j = 0; j < 31; j++) {
            if ((word >> j) & 1) {
                uint32_t absolute_pos = (i * 31) + j;
                temp_set[count++] = absolute_pos;
            }
        }
    }

    int* final_set = (int*)realloc(temp_set, count * sizeof(int));
    *set_size = count;
    return final_set;
}
