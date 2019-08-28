## source /opt/intel/compilers_and_libraries_2018/linux/mkl/bin/mklvars.sh intel64

MKLROOT=$(CURDIR)/../mklml
IOMP5_PATH=$(MKLROOT)/lib
MKL_LIB=$(MKLROOT)/lib

# for dynamic linking with MKL
CFLAGS=-I$(MKLROOT)/include -L${MKL_LIB} -Wl,--start-group -lmklml_intel -liomp5 -Wl,--end-group -lpthread -g -DMKL_ALIGN

# for static linking with MKL
## mkl= -I$(MKLROOT)/include -Wl,--start-group $(MKL_LIB)/libmkl_intel_lp64.a $(MKL_LIB)/libmkl_intel_thread.a $(MKL_LIB)/libmkl_core.a -Wl,--end-group -L$(IOMP5_PATH) -liomp5 -ldl -lpthread 

all: packed_sgemm_cblas_ALIGN_LD sgemm_cblas_ALIGN_LD

packed_sgemm_cblas_ALIGN_LD: packed_sgemm_cblas.cpp
	gcc $(CFLAGS) $< -o $@
sgemm_cblas_ALIGN_LD: sgemm_cblas.cpp
	gcc $(CFLAGS) $< -o $@

clean:
	rm packed_sgemm_cblas_ALIGN_LD
	rm sgemm_cblas_ALIGN_LD
