CC ?= gcc
CFLAGS ?= -O3 -march=native -Wall -Wextra -std=c11
LDLIBS ?= -lm

ORIG_SOURCES := \
	kernels/orig/constant_3d_tensor_vector.c \
	kernels/orig/constant_4d_tensor.c \
	kernels/orig/constant_attention_score.c \
	kernels/orig/constant_batched_gemm.c \
	kernels/orig/constant_context_lookup.c \
	kernels/orig/constant_matrix_matrix.c \
	kernels/orig/constant_matrix_vector.c \
	kernels/orig/constant_rowwise_softmax_max.c \
	kernels/orig/constant_stencil5pt.c

TILED_SOURCES := \
	kernels/tiled/tiled_3d_tensor_vector.c \
	kernels/tiled/tiled_4d_tensor.c \
	kernels/tiled/tiled_attention_score.c \
	kernels/tiled/tiled_batched_gemm.c \
	kernels/tiled/tiled_context_lookup.c \
	kernels/tiled/tiled_matrix_matrix.c \
	kernels/tiled/tiled_matrix_vector.c \
	kernels/tiled/tiled_rowwise_softmax_max.c

SOURCES := benchmark_runner.c $(ORIG_SOURCES) $(TILED_SOURCES)
TARGET := bin/benchmark_runner

.PHONY: all clean list

all: $(TARGET)

$(TARGET): $(SOURCES) include/kernel_interface.h ../pmc_measurement/pmc_l1d_misses.h | bin
	$(CC) $(CFLAGS) $(SOURCES) $(LDLIBS) -o $@

bin:
	mkdir -p $@

list: $(TARGET)
	$(TARGET) --list

clean:
	rm -f $(TARGET) ../data/pmu_results_17_new.csv ../data/salt_vs_hw_misses_results.csv ../output/salt_vs_hw_misses_new_pmc.svg
