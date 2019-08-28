source /opt/intel/compilers_and_libraries_2019.4.243/linux/mkl/bin/mklvars.sh intel64
unset MKL_CBWR
export MKL_CBWR=AVX2
export KMP_AFFINITY=compact,1,0,granularity=fine
export OMP_NUM_THREADS=12
# unset MKL_NUM_THREADS
export MKL_NUM_THREADS=12


taskset -c 0-11 numactl ./packed_sgemm_cblas_ALIGN_LD 1024 300 4616 1000 24 2.2 64
taskset -c 0-11 numactl ./packed_sgemm_cblas_ALIGN_LD 512 300 1024 1000 24 2.2 64
taskset -c 0-11 numactl ./packed_sgemm_cblas_ALIGN_LD 256 300 512 1000 24 2.2 64
taskset -c 0-11 numactl ./packed_sgemm_cblas_ALIGN_LD 1 300 256 1000 24 2.2 64


taskset -c 0-11 numactl ./sgemm_cblas_ALIGN_LD 1024 300 4616 1000 24 2.2 64
taskset -c 0-11 numactl ./sgemm_cblas_ALIGN_LD 512 300 1024 1000 24 2.2 64
taskset -c 0-11 numactl ./sgemm_cblas_ALIGN_LD 256 300 512 1000 24 2.2 64
taskset -c 0-11 numactl ./sgemm_cblas_ALIGN_LD 1 300 256 1000 24 2.2 64
