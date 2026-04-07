#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#define cudaCheckErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, const char *file, int line, int abort=1)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int      CHUNK_BITS = 16;
uint32_t CHUNK_MASK;
uint32_t CHUNK_COUNT;
uint32_t BITMASK_WORDS_PER_CHUNK;
uint32_t GLOBAL_BITMASK_WORDS;

static void recompute_constants(void)
{

    if (CHUNK_BITS < 1 || CHUNK_BITS > 31) {
        fprintf(stderr, "Error: CHUNK_BITS must be 1..31\n");
        exit(1);
    }

    uint32_t bits_per_chunk = 1u << CHUNK_BITS;

    uint32_t num_chunks_total = (CHUNK_BITS == 31) ? 2u : (1u << (32 - CHUNK_BITS));

    CHUNK_MASK               = bits_per_chunk - 1u;
    CHUNK_COUNT              = num_chunks_total; 
    BITMASK_WORDS_PER_CHUNK  = (bits_per_chunk + 31u) / 32u;
    GLOBAL_BITMASK_WORDS     = (num_chunks_total + 31u) / 32u;
}

struct BitEntry {
    int      chunk_idx;
    uint32_t bit_pos; 
};

__global__ void mark_present_chunks_shared_kernel(
    const int *d_setA_test,
    int setSize,
    uint32_t *d_chunk_bitmask,
    uint32_t global_bitmask_words,
    int chunk_bits,
    int use_shared) 
{
    extern __shared__ uint32_t s_bitmask[];

    if (use_shared) {
        for (int i = threadIdx.x; i < global_bitmask_words; i += blockDim.x)
            s_bitmask[i] = 0;
        __syncthreads();
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < setSize) {
        uint32_t value = (uint32_t)d_setA_test[idx];
        uint32_t chunk = value >> chunk_bits;

        size_t word_index = chunk / 32;
        if (word_index < global_bitmask_words) {
            int bit_position = chunk % 32;
            if (use_shared) {
                atomicOr(&s_bitmask[word_index], (1U << bit_position));
            } else {

                atomicOr(&d_chunk_bitmask[word_index], (1U << bit_position));
            }
        }
        idx += gridDim.x * blockDim.x;
    }

    if (use_shared) {
        __syncthreads();
        for (int i = threadIdx.x; i < global_bitmask_words; i += blockDim.x) {
            uint32_t w = s_bitmask[i];
            if (w) atomicOr(&d_chunk_bitmask[i], w);
        }
    }
}

struct count_set_bits_functor {
    __device__ int operator()(uint32_t word) const { return __popc(word); }
};

__global__ void populate_chunk_maps_kernel(
    const uint32_t *d_chunk_bitmask,
    const uint32_t *d_scan_results,
    int *d_chunk_to_index,
    uint32_t *d_index_to_chunk_value,
    uint32_t global_bitmask_words)
{
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (word_idx >= global_bitmask_words) return;

    uint32_t cur = d_chunk_bitmask[word_idx];
    uint32_t base = d_scan_results[word_idx];

    for (int j = 0; j < 32; ++j) {
        if ((cur >> j) & 1) {
            uint32_t chunk_val = (word_idx * 32) + j;
            int pop_before = __popc(cur & ((1U << j) - 1));
            int global_idx = base + pop_before;
            d_chunk_to_index[chunk_val] = global_idx;
            d_index_to_chunk_value[global_idx] = chunk_val;
        }
    }
}

__global__ void tag_elements_kernel(
    const int *d_setA_test,
    int setSize,
    const int *d_chunk_to_index,
    BitEntry *d_bit_entries,
    int chunk_bits,
    uint32_t chunk_mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < setSize) {
        uint32_t value = (uint32_t)d_setA_test[idx];
        uint32_t chunk_val = value >> chunk_bits;

        uint32_t bit_pos = value & chunk_mask; 

        d_bit_entries[idx].chunk_idx = d_chunk_to_index[chunk_val];
        d_bit_entries[idx].bit_pos   = bit_pos;

        idx += gridDim.x * blockDim.x;
    }
}

__global__ void count_elements_per_chunk_kernel(
    const BitEntry *d_bit_entries,
    int setSize,
    uint32_t *d_local_chunk_counts,
    uint32_t total_chunks_A_GPU,
    int use_shared)
{
    extern __shared__ uint32_t s_chunk_counts[];
    uint32_t *block_counts = d_local_chunk_counts + (blockIdx.x * total_chunks_A_GPU);

    if (use_shared) {
        for (int i = threadIdx.x; i < total_chunks_A_GPU; i += blockDim.x) s_chunk_counts[i] = 0;
    } else {
        for (int i = threadIdx.x; i < total_chunks_A_GPU; i += blockDim.x) block_counts[i] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < setSize) {
        int c = d_bit_entries[idx].chunk_idx;
        if (use_shared) atomicAdd(&s_chunk_counts[c], 1);
        else            atomicAdd(&block_counts[c], 1);
        idx += gridDim.x * blockDim.x;
    }
    __syncthreads();

    if (use_shared) {
        for (int i = threadIdx.x; i < total_chunks_A_GPU; i += blockDim.x)
            d_local_chunk_counts[blockIdx.x * total_chunks_A_GPU + i] = s_chunk_counts[i];
    }
}

__global__ void compute_prefix_sums_kernel(
    const uint32_t *d_local_chunk_counts,
    int num_binning_tiles,
    uint32_t total_chunks_A_GPU,
    uint32_t *d_tile_offsets,
    uint32_t *d_chunk_counts)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= total_chunks_A_GPU) return;

    uint32_t sum = 0;
    for (int t = 0; t < num_binning_tiles; ++t) {
        uint32_t cnt = d_local_chunk_counts[t * total_chunks_A_GPU + c];
        d_tile_offsets[t * total_chunks_A_GPU + c] = sum;
        sum += cnt;
    }
    d_chunk_counts[c] = sum;
}

__global__ void bin_elements_kernel(
    const BitEntry *d_bit_entries,
    int setSize,
    const uint32_t *d_chunk_offsets,
    const uint32_t *d_tile_offsets,
    BitEntry *d_sorted_bit_entries,
    uint32_t total_chunks_A_GPU,
    int use_shared,
    uint32_t *d_fallback_local_counters)
{
    extern __shared__ uint32_t s_local_counters[];
    uint32_t *block_counters = d_fallback_local_counters ?
                               d_fallback_local_counters + blockIdx.x * total_chunks_A_GPU : NULL;

    if (use_shared) {
        for (int i = threadIdx.x; i < total_chunks_A_GPU; i += blockDim.x) s_local_counters[i] = 0;
    } else if (block_counters) {
        for (int i = threadIdx.x; i < total_chunks_A_GPU; i += blockDim.x) block_counters[i] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < setSize) {
        BitEntry e = d_bit_entries[idx];
        uint32_t local_pos;
        if (use_shared) local_pos = atomicAdd(&s_local_counters[e.chunk_idx], 1);
        else            local_pos = atomicAdd(&block_counters[e.chunk_idx], 1);

        uint32_t global_pos = d_chunk_offsets[e.chunk_idx] +
                              d_tile_offsets[blockIdx.x * total_chunks_A_GPU + e.chunk_idx] +
                              local_pos;
        d_sorted_bit_entries[global_pos] = e;
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void populate_hipes_bitmap_kernel_chunk_centric(
    const BitEntry *d_sorted_bit_entries,
    const uint32_t *d_chunk_offsets,
    uint32_t total_chunks_A_GPU,
    uint32_t *d_setA_test_GPU_hipes_bitmap,
    const uint32_t *d_index_to_chunk_value,
    int setSize,
    uint32_t bitmask_words_per_chunk,
    int use_shared)
{
    uint32_t chunk_global_index = blockIdx.x;
    if (chunk_global_index >= total_chunks_A_GPU) return;

    uint32_t current_chunk_value = d_index_to_chunk_value[chunk_global_index];

    size_t part_sz = 1 + bitmask_words_per_chunk;
    size_t off = chunk_global_index * part_sz;

    extern __shared__ uint32_t s_chunk_bitmask[];
    uint32_t* p_target_bitmask;

    if (use_shared) {
        p_target_bitmask = s_chunk_bitmask;
    } else {
        p_target_bitmask = &d_setA_test_GPU_hipes_bitmap[off + 1];
    }

    for (int i = threadIdx.x; i < bitmask_words_per_chunk; i += blockDim.x)
        p_target_bitmask[i] = 0;
    __syncthreads();

    uint32_t start_idx = d_chunk_offsets[chunk_global_index];
    uint32_t end_idx   = (chunk_global_index + 1 < total_chunks_A_GPU) ?
                         d_chunk_offsets[chunk_global_index + 1] : setSize;

    for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {

        uint32_t bit_position = d_sorted_bit_entries[i].bit_pos;

        size_t word_idx = bit_position / 32;
        int bit_in_word = bit_position % 32;
        if (word_idx < bitmask_words_per_chunk) {
            atomicOr(&p_target_bitmask[word_idx], (1U << bit_in_word));
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        d_setA_test_GPU_hipes_bitmap[off] = current_chunk_value;
        if (use_shared) {
            for (int i = 0; i < bitmask_words_per_chunk; ++i)
                d_setA_test_GPU_hipes_bitmap[off + 1 + i] = s_chunk_bitmask[i];
        }
    }
}

static void compute_chunks_and_bitmask(const int *set, size_t n,
                                      uint32_t *total_chunks, uint8_t **bitmask)
{
    size_t bytes = (CHUNK_COUNT + 7) / 8;
    *bitmask = (uint8_t*)calloc(bytes, 1);
    *total_chunks = 0;

    for (size_t i = 0; i < n; ++i) {
        uint32_t v = (uint32_t)set[i];
        uint32_t chunk = v >> CHUNK_BITS;

        uint32_t byte = chunk / 8; 
        uint8_t  bit  = chunk % 8;

        if (!((*bitmask)[byte] & (1 << bit))) {
            (*bitmask)[byte] |= (1 << bit);
            ++(*total_chunks);
        }
    }
}

static uint8_t* create_hipes_bitmap(const int *set, size_t n,
                                   uint32_t total_chunks, uint8_t *chunk_bitmask)
{

    size_t global_index_bytes = (CHUNK_COUNT + 7) / 8;

    size_t payload_bytes = ((1u << CHUNK_BITS) + 7) / 8;

    size_t sz = total_chunks * (2 + payload_bytes);
    uint8_t *hipes = (uint8_t*)malloc(sz);
    memset(hipes, 0, sz);

    size_t off = 0;

    for (size_t i = 0; i < global_index_bytes; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (chunk_bitmask[i] & (1 << j)) {
                uint32_t chunk_id = i * 8 + j;

                hipes[off++] = (chunk_id >> 8) & 0xFF;
                hipes[off++] = chunk_id & 0xFF;

                uint8_t *ba = hipes + off;
                off += payload_bytes; 

                for (size_t k = 0; k < n; ++k) {
                    uint32_t v = (uint32_t)set[k];
                    if ((v >> CHUNK_BITS) == chunk_id) {
                        uint32_t pos = v & CHUNK_MASK;
                        ba[pos / 8] |= (1 << (pos % 8));
                    }
                }
            }
        }
    }
    return hipes;
}

static int* hipes_bitmap_to_set(const uint8_t *hipes, uint32_t total_chunks,
                               size_t *out_size)
{
    size_t payload_bytes = ((1u << CHUNK_BITS) + 7) / 8;

    size_t cap = 0;
    size_t off = 0;
    for (uint32_t c = 0; c < total_chunks; ++c) {
        off += 2; 
        const uint8_t *ba = hipes + off;
        for (size_t i = 0; i < payload_bytes; ++i) {
            uint8_t b = ba[i];
            while (b) { cap += b & 1; b >>= 1; }
        }
        off += payload_bytes;
    }

    int *res = (int*)malloc(cap * sizeof(int));
    if (!res) { *out_size = 0; return NULL; }

    off = 0; size_t idx = 0;
    for (uint32_t c = 0; c < total_chunks; ++c) {
        uint32_t chunk_id = (hipes[off] << 8) | hipes[off + 1];
        off += 2;
        const uint8_t *ba = hipes + off;
        off += payload_bytes;

        for (size_t i = 0; i < payload_bytes; ++i) {
            uint8_t b = ba[i];
            for (int bit = 0; bit < 8; ++bit) {
                if (b & (1 << bit)) {
                    uint32_t low = i * 8 + bit;
                    res[idx++] = (int)((chunk_id << CHUNK_BITS) | low);
                }
            }
        }
    }
    *out_size = idx;
    return res;
}

int main(int argc, char **argv)
{

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <input_file> [chunk_bits]\n", argv[0]);
        return 1;
    }
    const char *filename = argv[1];
    if (argc == 3) CHUNK_BITS = atoi(argv[2]);
    recompute_constants();

    FILE *f = fopen(filename, "r");
    if (!f) { perror("fopen"); return 1; }

    int *numbers = NULL;
    size_t capacity = 0, setSize = 0;
    int val;
    while (fscanf(f, "%d", &val) == 1) {
        if (setSize == capacity) {
            capacity = capacity ? capacity * 2 : 1024;
            numbers = (int*)realloc(numbers, capacity * sizeof(int));
        }
        numbers[setSize++] = val;
    }
    fclose(f);
    printf("Read %zu numbers, chunk_bits=%d\n", setSize, CHUNK_BITS);

    cudaEvent_t start, stop;
    cudaCheckErrors(cudaEventCreate(&start));
    cudaCheckErrors(cudaEventCreate(&stop));
    cudaCheckErrors(cudaEventRecord(start));

    int *d_setA_test;
    uint32_t *d_chunk_bitmask;
    int      *d_chunk_to_index;
    uint32_t *d_index_to_chunk_value;

    cudaCheckErrors(cudaMalloc(&d_setA_test,               setSize * sizeof(int)));
    cudaCheckErrors(cudaMalloc(&d_chunk_bitmask,           GLOBAL_BITMASK_WORDS * sizeof(uint32_t)));
    cudaCheckErrors(cudaMalloc(&d_chunk_to_index,          CHUNK_COUNT * sizeof(int)));
    cudaCheckErrors(cudaMalloc(&d_index_to_chunk_value,    CHUNK_COUNT * sizeof(uint32_t)));

    cudaCheckErrors(cudaMemcpy(d_setA_test, numbers, setSize * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemset(d_chunk_bitmask, 0, GLOBAL_BITMASK_WORDS * sizeof(uint32_t)));
    cudaCheckErrors(cudaMemset(d_chunk_to_index, -1, CHUNK_COUNT * sizeof(int)));

    cudaStream_t thrust_stream;
    cudaCheckErrors(cudaStreamCreate(&thrust_stream));

    const int BLOCKS0 = 256, TPB0 = 256;

    int use_shared0 = (GLOBAL_BITMASK_WORDS * sizeof(uint32_t) <= 48 * 1024);
    size_t shmem0   = use_shared0 ? (GLOBAL_BITMASK_WORDS * sizeof(uint32_t)) : 0;
    mark_present_chunks_shared_kernel<<<BLOCKS0, TPB0, shmem0>>>(
        d_setA_test, (int)setSize, d_chunk_bitmask,
        GLOBAL_BITMASK_WORDS, CHUNK_BITS, use_shared0); 
    cudaCheckErrors(cudaDeviceSynchronize());

    thrust::device_vector<uint32_t> d_bitmask_vec(d_chunk_bitmask,
                                                  d_chunk_bitmask + GLOBAL_BITMASK_WORDS);
    thrust::device_vector<uint32_t> d_scan_input(GLOBAL_BITMASK_WORDS);
    thrust::device_vector<uint32_t> d_scan_result(GLOBAL_BITMASK_WORDS);

    thrust::transform(thrust::cuda::par.on(thrust_stream),
                      d_bitmask_vec.begin(), d_bitmask_vec.end(),
                      d_scan_input.begin(), count_set_bits_functor());

    thrust::exclusive_scan(thrust::cuda::par.on(thrust_stream),
                           d_scan_input.begin(), d_scan_input.end(),
                           d_scan_result.begin());
    cudaCheckErrors(cudaStreamSynchronize(thrust_stream));

    uint32_t total_chunks_A_GPU;
    {
        uint32_t last_word, last_popc, last_scan;
        cudaCheckErrors(cudaMemcpy(&last_word,
                                   d_chunk_bitmask + GLOBAL_BITMASK_WORDS - 1,
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost));
        last_popc = __builtin_popcount(last_word);
        cudaCheckErrors(cudaMemcpy(&last_scan,
                                   thrust::raw_pointer_cast(d_scan_result.data()) + GLOBAL_BITMASK_WORDS - 1,
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost));
        total_chunks_A_GPU = last_scan + last_popc;
    }

    const int TPB_MAP = 256;
    const int BLK_MAP = (GLOBAL_BITMASK_WORDS + TPB_MAP - 1) / TPB_MAP;
    populate_chunk_maps_kernel<<<BLK_MAP, TPB_MAP>>>(
        d_chunk_bitmask,
        thrust::raw_pointer_cast(d_scan_result.data()),
        d_chunk_to_index,
        d_index_to_chunk_value,
        GLOBAL_BITMASK_WORDS);
    cudaCheckErrors(cudaDeviceSynchronize());

    cudaCheckErrors(cudaFree(d_chunk_bitmask));

    BitEntry *d_bit_entries;
    cudaCheckErrors(cudaMalloc(&d_bit_entries, setSize * sizeof(BitEntry)));

    const int TPB2 = 256;
    const int BLK2 = (setSize + TPB2 - 1) / TPB2;
    tag_elements_kernel<<<BLK2, TPB2>>>(
        d_setA_test, (int)setSize, d_chunk_to_index, d_bit_entries,
        CHUNK_BITS, CHUNK_MASK);
    cudaCheckErrors(cudaDeviceSynchronize());

    cudaCheckErrors(cudaFree(d_setA_test));
    cudaCheckErrors(cudaFree(d_chunk_to_index));

    const int NUM_TILES = 256;
    uint32_t *d_local_chunk_counts;
    cudaCheckErrors(cudaMalloc(&d_local_chunk_counts,
                               NUM_TILES * total_chunks_A_GPU * sizeof(uint32_t)));
    cudaCheckErrors(cudaMemset(d_local_chunk_counts, 0,
                               NUM_TILES * total_chunks_A_GPU * sizeof(uint32_t)));

    int use_shared_cnt = (total_chunks_A_GPU * sizeof(uint32_t) <= 48*1024);
    size_t shmem_cnt = use_shared_cnt ? total_chunks_A_GPU * sizeof(uint32_t) : 0;
    count_elements_per_chunk_kernel<<<NUM_TILES, TPB2, shmem_cnt>>>(
        d_bit_entries, (int)setSize, d_local_chunk_counts,
        total_chunks_A_GPU, use_shared_cnt);
    cudaCheckErrors(cudaDeviceSynchronize());

    uint32_t *d_tile_offsets, *d_chunk_counts, *d_chunk_offsets;
    cudaCheckErrors(cudaMalloc(&d_tile_offsets,
                               NUM_TILES * total_chunks_A_GPU * sizeof(uint32_t)));
    cudaCheckErrors(cudaMalloc(&d_chunk_counts, total_chunks_A_GPU * sizeof(uint32_t)));
    cudaCheckErrors(cudaMalloc(&d_chunk_offsets, (total_chunks_A_GPU + 1) * sizeof(uint32_t)));

    const int TPB_PREFIX = 256;
    const int BLK_PREFIX = (total_chunks_A_GPU + TPB_PREFIX - 1) / TPB_PREFIX;
    compute_prefix_sums_kernel<<<BLK_PREFIX, TPB_PREFIX>>>(
        d_local_chunk_counts, NUM_TILES, total_chunks_A_GPU,
        d_tile_offsets, d_chunk_counts);
    cudaCheckErrors(cudaDeviceSynchronize());

    thrust::device_vector<uint32_t> d_chunk_counts_vec(d_chunk_counts,
                                                       d_chunk_counts + total_chunks_A_GPU);
    thrust::device_vector<uint32_t> d_chunk_offsets_vec(total_chunks_A_GPU + 1);
    thrust::exclusive_scan(thrust::cuda::par.on(thrust_stream),
                           d_chunk_counts_vec.begin(),
                           d_chunk_counts_vec.end(),
                           d_chunk_offsets_vec.begin());
    cudaCheckErrors(cudaStreamSynchronize(thrust_stream));
    cudaCheckErrors(cudaMemcpy(d_chunk_offsets,
                               thrust::raw_pointer_cast(d_chunk_offsets_vec.data()),
                               (total_chunks_A_GPU + 1) * sizeof(uint32_t),
                               cudaMemcpyDeviceToDevice));
    cudaCheckErrors(cudaMemcpy(d_chunk_offsets + total_chunks_A_GPU,
                               &setSize, sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaFree(d_chunk_counts));
    cudaCheckErrors(cudaFree(d_local_chunk_counts));

    BitEntry *d_sorted_bit_entries;
    cudaCheckErrors(cudaMalloc(&d_sorted_bit_entries, setSize * sizeof(BitEntry)));

    int use_shared_scat = (total_chunks_A_GPU * sizeof(uint32_t) <= 48*1024);
    size_t shmem_scat = use_shared_scat ? total_chunks_A_GPU * sizeof(uint32_t) : 0;
    uint32_t *d_fallback = NULL;
    if (!use_shared_scat) {
        cudaCheckErrors(cudaMalloc(&d_fallback,
                                   NUM_TILES * total_chunks_A_GPU * sizeof(uint32_t)));
        cudaCheckErrors(cudaMemset(d_fallback, 0,
                                   NUM_TILES * total_chunks_A_GPU * sizeof(uint32_t)));
    }

    bin_elements_kernel<<<NUM_TILES, TPB2, shmem_scat>>>(
        d_bit_entries, (int)setSize, d_chunk_offsets,
        d_tile_offsets, d_sorted_bit_entries,
        total_chunks_A_GPU, use_shared_scat, d_fallback);
    cudaCheckErrors(cudaDeviceSynchronize());

    if (d_fallback) cudaCheckErrors(cudaFree(d_fallback));
    cudaCheckErrors(cudaFree(d_bit_entries));
    cudaCheckErrors(cudaFree(d_tile_offsets));

    size_t hipes_words = total_chunks_A_GPU * (1 + BITMASK_WORDS_PER_CHUNK);
    uint32_t *d_setA_test_GPU_hipes_bitmap;
    cudaCheckErrors(cudaMalloc(&d_setA_test_GPU_hipes_bitmap,
                               hipes_words * sizeof(uint32_t)));
    cudaCheckErrors(cudaMemset(d_setA_test_GPU_hipes_bitmap, 0,
                               hipes_words * sizeof(uint32_t)));

    int use_shared4 = (BITMASK_WORDS_PER_CHUNK * sizeof(uint32_t) <= 48 * 1024);
    size_t shmem_final = use_shared4 ? (BITMASK_WORDS_PER_CHUNK * sizeof(uint32_t)) : 0;

    populate_hipes_bitmap_kernel_chunk_centric<<<total_chunks_A_GPU, TPB2, shmem_final>>>(
        d_sorted_bit_entries, d_chunk_offsets, total_chunks_A_GPU,
        d_setA_test_GPU_hipes_bitmap, d_index_to_chunk_value,
        (int)setSize, BITMASK_WORDS_PER_CHUNK, use_shared4); 
    cudaCheckErrors(cudaDeviceSynchronize());

    cudaCheckErrors(cudaEventRecord(stop));
    cudaCheckErrors(cudaEventSynchronize(stop));
    float ms;
    cudaCheckErrors(cudaEventElapsedTime(&ms, start, stop));

    size_t initial_size = setSize * sizeof(int);
    size_t compressed_size = hipes_words * sizeof(uint32_t);
    double compression_speed_mbs = (ms > 0) ? (initial_size / (1024.0 * 1024.0)) / (ms / 1000.0) : 0;
    double effectiveness_bits_per_byte = (compressed_size > 0 && initial_size > 0) ? (double)(compressed_size * 8.0) / initial_size : 0;

    printf("Initial size (uncompressed): %zu bytes\n", initial_size);
    printf("Final size (compressed): %zu bytes\n", compressed_size);
    printf("Compression Ratio (Uncompressed/Compressed): %.2f : 1\n", (double)initial_size / compressed_size);
    printf("Effectiveness (Bits Per Byte): %.2f\n", effectiveness_bits_per_byte);
    printf("Compression Speed: %.2f MB/s\n", compression_speed_mbs);
    printf("Space reduction: %.2f%%\n", (1.0 - (double)compressed_size / initial_size) * 100);

    return 0;
}
