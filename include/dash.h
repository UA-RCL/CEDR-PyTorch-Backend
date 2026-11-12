#pragma once

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include "dash_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void DASH_ZIP_flt(dash_cmplx_flt_type* input_1, dash_cmplx_flt_type* input_2, dash_cmplx_flt_type* output, size_t size, zip_op_t op);
void DASH_ZIP_flt_nb(dash_cmplx_flt_type** input_1, dash_cmplx_flt_type** input_2, dash_cmplx_flt_type** output, size_t* size, zip_op_t* op, cedr_barrier_t* kernel_barrier);

void DASH_ZIP_int(dash_cmplx_int_type* input_1, dash_cmplx_int_type* input_2, dash_cmplx_int_type* output, size_t size, zip_op_t op);
void DASH_ZIP_int_nb(dash_cmplx_int_type** input_1, dash_cmplx_int_type** input_2, dash_cmplx_int_type** output, size_t* size, zip_op_t* op, cedr_barrier_t* kernel_barrier);

void DASH_GEMM_flt(dash_cmplx_flt_type* A, dash_cmplx_flt_type* B, dash_cmplx_flt_type* C, size_t Row_A, size_t Col_A, size_t Col_B);
void DASH_GEMM_flt_nb(dash_cmplx_flt_type** A, dash_cmplx_flt_type** B, dash_cmplx_flt_type** C, size_t* Row_A, size_t* Col_A, size_t* Col_B, cedr_barrier_t* kernel_barrier);

void DASH_GEMM_int(dash_cmplx_int_type* A, dash_cmplx_int_type* B, dash_cmplx_int_type* C, size_t Row_A, size_t Col_A, size_t Col_B);
void DASH_GEMM_int_nb(dash_cmplx_int_type** A, dash_cmplx_int_type** B, dash_cmplx_int_type** C, size_t* Row_A, size_t* Col_A, size_t* Col_B, cedr_barrier_t* kernel_barrier);

void DASH_CONV_2D_flt(dash_re_flt_type *input, int height, int width, dash_re_flt_type *mask, int mask_size, dash_re_flt_type *output);
void DASH_CONV_2D_flt_nb(dash_re_flt_type **input, int *height, int *width, dash_re_flt_type **mask, int *mask_size, dash_re_flt_type **output, cedr_barrier_t* kernel_barrier);

void DASH_CONV_2D_int(dash_re_int_type *input, int height, int width, dash_re_flt_type *mask, int mask_size, dash_re_int_type *output);
void DASH_CONV_2D_int_nb(dash_re_int_type **input, int *height, int *width, dash_re_flt_type **mask, int *mask_size, dash_re_int_type **output, cedr_barrier_t* kernel_barrier);

void DASH_CONV_1D_flt(dash_re_flt_type* input, int size, dash_re_flt_type* mask, int mask_size, dash_re_flt_type* output);
void DASH_CONV_1D_flt_nb(dash_re_flt_type** input, int* size, dash_re_flt_type** mask, int* mask_size, dash_re_flt_type** output, cedr_barrier_t* kernel_barrier);

#ifdef __cplusplus
} // Close 'extern "C"'
#endif
