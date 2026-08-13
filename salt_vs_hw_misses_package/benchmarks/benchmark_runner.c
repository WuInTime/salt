#include "include/kernel_interface.h"
#include "../pmc_measurement/pmc_l1d_misses.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

extern kernel_t constant_3d_tensor_vector_kernel;
extern kernel_t constant_4d_tensor_kernel;
extern kernel_t constant_attention_score_kernel;
extern kernel_t constant_batched_gemm_kernel;
extern kernel_t constant_context_lookup_kernel;
extern kernel_t constant_matrix_matrix_kernel;
extern kernel_t constant_matrix_vector_kernel;
extern kernel_t constant_rowwise_softmax_max_kernel;
extern kernel_t constant_stencil5pt_kernel;
extern kernel_t tiled_3d_tensor_vector_kernel;
extern kernel_t tiled_4d_tensor_kernel;
extern kernel_t tiled_attention_score_kernel;
extern kernel_t tiled_batched_gemm_kernel;
extern kernel_t tiled_context_lookup_kernel;
extern kernel_t tiled_matrix_matrix_kernel;
extern kernel_t tiled_matrix_vector_kernel;
extern kernel_t tiled_rowwise_softmax_max_kernel;

typedef struct {
    const char *display_name;
    kernel_t *kernel;
} benchmark_entry_t;

static benchmark_entry_t benchmarks[] = {
    {"orig_3d_tensor_vector", &constant_3d_tensor_vector_kernel},
    {"orig_4d_tensor", &constant_4d_tensor_kernel},
    {"orig_attention_score", &constant_attention_score_kernel},
    {"orig_batched_gemm", &constant_batched_gemm_kernel},
    {"orig_context_lookup", &constant_context_lookup_kernel},
    {"orig_matrix_matrix", &constant_matrix_matrix_kernel},
    {"orig_matrix_vector", &constant_matrix_vector_kernel},
    {"orig_rowwise_softmax_max", &constant_rowwise_softmax_max_kernel},
    {"tiled_3d_tensor_vector", &tiled_3d_tensor_vector_kernel},
    {"tiled_4d_tensor", &tiled_4d_tensor_kernel},
    {"tiled_attention_score", &tiled_attention_score_kernel},
    {"tiled_batched_gemm", &tiled_batched_gemm_kernel},
    {"tiled_context_lookup", &tiled_context_lookup_kernel},
    {"tiled_matrix_matrix", &tiled_matrix_matrix_kernel},
    {"tiled_matrix_vector", &tiled_matrix_vector_kernel},
    {"tiled_rowwise_softmax_max", &tiled_rowwise_softmax_max_kernel},
    {"orig_stencil", &constant_stencil5pt_kernel},
};

static const size_t benchmark_count = sizeof(benchmarks) / sizeof(benchmarks[0]);

static benchmark_entry_t *find_benchmark(const char *name)
{
    for (size_t i = 0; i < benchmark_count; ++i) {
        if (strcmp(name, benchmarks[i].display_name) == 0 ||
            strcmp(name, benchmarks[i].kernel->name) == 0) {
            return &benchmarks[i];
        }
    }
    return NULL;
}

static void print_names(FILE *stream)
{
    for (size_t i = 0; i < benchmark_count; ++i) {
        fprintf(stream, "%s\n", benchmarks[i].display_name);
    }
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--list") == 0) {
        print_names(stdout);
        return 0;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: %s BENCHMARK\navailable benchmarks:\n", argv[0]);
        print_names(stderr);
        return 2;
    }

    benchmark_entry_t *entry = find_benchmark(argv[1]);
    if (entry == NULL) {
        fprintf(stderr, "unknown benchmark: %s\navailable benchmarks:\n", argv[1]);
        print_names(stderr);
        return 2;
    }

    kernel_t *kernel = entry->kernel;
    kernel->init();

    pmc_l1d_misses_t counter = {.fd = -1};
    if (pmc_l1d_misses_open(&counter) != 0) {
        kernel->cleanup();
        return 3;
    }
    if (pmc_l1d_misses_start(&counter) != 0) {
        pmc_l1d_misses_close(&counter);
        kernel->cleanup();
        return 3;
    }

    kernel->execute();

    uint64_t misses = 0;
    if (pmc_l1d_misses_stop(&counter, &misses) != 0) {
        pmc_l1d_misses_close(&counter);
        kernel->cleanup();
        return 3;
    }
    pmc_l1d_misses_close(&counter);
    kernel->cleanup();

    printf("%" PRIu64 "\n", misses);
    return 0;
}
