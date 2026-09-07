NVCC    := nvcc
ARCH    := -arch=sm_60
CFLAGS  := -O3 -std=c++17 $(ARCH) -lineinfo -Xptxas -v
LIBS    := -lcublas

BINS := bench_cublas gemm attention bwd

all: $(BINS)

bench_cublas: bench_cublas.cu
	$(NVCC) $(CFLAGS) -o $@ $< $(LIBS) 2> $@.ptxas.txt || (cat $@.ptxas.txt; false)
	@grep -E 'registers|smem|Function properties|Compiling entry' $@.ptxas.txt || true

clean:
	rm -f $(BINS) *.ptxas.txt

gemm: gemm.cu common.cuh
	$(NVCC) $(CFLAGS) -o $@ $< $(LIBS) 2> $@.ptxas.txt || (cat $@.ptxas.txt; false)
	@grep -E 'Function properties|registers|smem' $@.ptxas.txt || true

attention: attention.cu flash_fwd.cuh common.cuh
	$(NVCC) $(CFLAGS) -o $@ $< $(LIBS) 2> $@.ptxas.txt || (cat $@.ptxas.txt; false)
	@grep -E 'fwd_kernel|registers|smem' $@.ptxas.txt || true

bwd: bwd.cu flash_bwd.cuh flash_fwd.cuh common.cuh
	$(NVCC) $(CFLAGS) -o $@ $< $(LIBS) 2> $@.ptxas.txt || (cat $@.ptxas.txt; false)
	@grep -E 'bwd_kernel|registers|smem' $@.ptxas.txt || true
