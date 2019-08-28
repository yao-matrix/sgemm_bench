/*******************************************************************************
!   Copyright(C) 2010-2017 Intel Corporation. All Rights Reserved.
!   
!   The source code, information  and  material ("Material") contained herein is
!   owned  by Intel Corporation or its suppliers or licensors, and title to such
!   Material remains  with Intel Corporation  or its suppliers or licensors. The
!   Material  contains proprietary information  of  Intel or  its  suppliers and
!   licensors. The  Material is protected by worldwide copyright laws and treaty
!   provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
!   modified, published, uploaded, posted, transmitted, distributed or disclosed
!   in any way  without Intel's  prior  express written  permission. No  license
!   under  any patent, copyright  or  other intellectual property rights  in the
!   Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
!   implication, inducement,  estoppel or  otherwise.  Any  license  under  such
!   intellectual  property  rights must  be express  and  approved  by  Intel in
!   writing.
!   
!   *Third Party trademarks are the property of their respective owners.
!   
!   Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
!   this  notice or  any other notice embedded  in Materials by Intel or Intel's
!   suppliers or licensors in any way.
!
!*******************************************************************************
!  Content:
!      Performance Measurement Example Program ( sgemm )
!******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "mkl_cblas.h"
#include "mkl_service.h"
//#include <omp.h>
#include <sys/time.h>


// #define MKL_ALIGN
// #define MKL_LD

#define FLOAT_TYPE float
#define DATA_LEN 524288
#define MKL_MEM_ALIGNMENT (1024)
#define FIX_LD(x)   (((x + 255) & ~255) + 16)

#ifdef MKL_dsecnd
#include <omp.h>
#include <mkl.h>
#endif

int main(int argc, char* argv[]) {
    FLOAT_TYPE* A;
    FLOAT_TYPE* A_PACK = nullptr;
    FLOAT_TYPE* B;
    FLOAT_TYPE* C;
    FLOAT_TYPE  alpha = 1.0, beta = 0.0;
    CBLAS_ORDER    layout;
    CBLAS_TRANSPOSE transB;
    CBLAS_STORAGE transA;

    layout = CblasRowMajor;
    transA = CblasPacked; 
    transB = CblasNoTrans;

    int m = 1024;
    int n = 300;
    int k = 4616;


    int LOOP_COUNT = 1000;
    int i, cores = 16;
    int fpc = 64;      // default skx 2.1GHz, AVX512 FMA enabled. fpc: float operating per cycle
    float hz = 2.1;    // CPU frequency

    if (argc == 1) { 
        printf("Usage: %s m n k loop cores hz fpc\n", argv[0]);
        printf("\t hz: CPU frequency in GHz, default=2.1\n");
        printf("\t fpc: float ops per cycle, skx=64,hsw=32, snb/ivb=16, default=64\n");
    }

    if (argc >= 2) {
        m = atoi(argv[1]);
    }
    if (argc >= 3) {
        n = atoi(argv[2]);
    }
    if (argc >= 4) {
        k = atoi(argv[3]);
    }
    if (argc >= 5) {
        LOOP_COUNT = atoi(argv[4]);
    }
    if (argc >= 6) {
        cores = atoi(argv[5]);
    }
    if (argc >= 7) {
        hz = (float)atof(argv[6]);
    }
    if (argc >= 8) {
        fpc = atoi(argv[7]);
    }

    int lda = k;
    int ldb = n;
    int ldc = n;
#ifdef MKL_LD
    lda = FIX_LD(k);
    // ldb = FIX_LD(n);
    // ldc = FIX_LD(n);
#endif

#ifdef MKL_ALIGN
    A = (FLOAT_TYPE*)mkl_malloc(sizeof(FLOAT_TYPE)*lda*m, MKL_MEM_ALIGNMENT);
    B = (FLOAT_TYPE*)mkl_malloc(sizeof(FLOAT_TYPE)*ldb*k, MKL_MEM_ALIGNMENT);
    C = (FLOAT_TYPE*)mkl_malloc(sizeof(FLOAT_TYPE)*ldc*m, MKL_MEM_ALIGNMENT);
#else	
    A = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE)*lda*m);//, MKL_MEM_ALIGNMENT);
    B = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE)*ldb*k);//, MKL_MEM_ALIGNMENT);
    C = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE)*ldc*m); //, MKL_MEM_ALIGNMENT);
#endif

    printf("Size of Matrix A(m x k): %d x %d  lda = %d\n", m, k, lda);
    printf("Size of Matrix B(k x n): %d x %d  ldb = %d\n", k, n, ldb);
    printf("Size of Matrix C(m x n): %d x %d  ldc = %d\n", m, n, ldc);
    printf("LOOP COUNT           : %d \n", LOOP_COUNT);

    // fill A, B, C with random values
    for (i = 0; i < lda * m; ++i) {
        A[i] = (FLOAT_TYPE)rand() / RAND_MAX - 0.5;
    }

    for (i = 0; i < ldb * k; ++i) {
        B[i] = (FLOAT_TYPE)rand() / RAND_MAX - 0.5;
    }

    for (i = 0; i < ldc * m; ++i) {
        C[i] = (FLOAT_TYPE)rand() / RAND_MAX - 0.5;
    }
    
    if (A_PACK) {
        cblas_sgemm_free(A_PACK);
        A_PACK = nullptr;
    }
    A_PACK = cblas_sgemm_alloc(CblasAMatrix, m, n, k);
    if (!A_PACK) {
        printf("cannot alloc A_PACK for test");
        return -1;
    }
    cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, alpha, A, k, A_PACK);

    for (i = 0; i < 20; i++) {
        cblas_sgemm_compute(layout, transA, transB, m, n, k,  A_PACK, lda, B, ldb, beta, C, ldc);
    }

    struct  timeval start_time, end_time;
    gettimeofday(&start_time,NULL);     
    for (i = 0; i < LOOP_COUNT; ++i) {
        cblas_sgemm_compute(layout, transA, transB, m, n, k, A_PACK, lda, B, ldb, beta, C, ldc);
    }
    gettimeofday(&end_time,NULL);
    double end_usecs = end_time.tv_usec;
    double end_secs = end_time.tv_sec;

    double start_usecs = start_time.tv_usec;
    double start_secs = start_time.tv_sec;

    double total_time_us = ((end_secs - start_secs) * 1000000) + (end_usecs - start_usecs);
    // double avg_time_ms = total_time_us/1000/(double)LOOP_COUNT;
    double avg_secs = total_time_us * 1E-6 / LOOP_COUNT;
    
    // double gflops_per_sec = gflops/avg_secs;
    // time_end = dsecnd();
    // time_avg = (time_end - time_st)/LOOP_COUNT;
	
    double gflop = 2.0 * m * n * k * 1E-9;
    double pGflops = hz * fpc * cores;
    printf ("m = %d, n = %d, k = %d cores = %d gflops = %.5f peak = %.5f efficiency = %.2f%%\n", m, n, k, cores, gflop / avg_secs, pGflops, gflop * 100 / avg_secs / pGflops);
  
    printf("Average time: %lf ms \n", avg_secs * 1000);
    printf("GFlop       : %.5f \n", gflop);
    printf("GFlop/sec   : %.5f GFlops \n", gflop/avg_secs);

    if (A_PACK) {
        cblas_sgemm_free(A_PACK);
        A_PACK = nullptr;
    }

#ifdef MKL_ALIGN
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
#else
    free(A);
    free(B);
    free(C);
#endif
    A = nullptr;
    B = nullptr;
    C = nullptr;

    return 0;
}
