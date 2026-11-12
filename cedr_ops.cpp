#include <iostream>
#include <omp.h>
#include <dlfcn.h>
#include <cmath>

#include "include/dash.h"

extern void enqueue_kernel(const char* kernel_name, const char* precision_name, unsigned int n_vargs, ...);

static void (*DASH_CONV_2D_flt_nb_ptr)(dash_re_flt_type*, int, int, dash_re_flt_type*, int, dash_re_flt_type*, cedr_barrier_t*);
static void (*DASH_ZIP_flt_nb_ptr)(dash_cmplx_flt_type*, dash_cmplx_flt_type*, dash_cmplx_flt_type*, size_t, zip_op_t);
static void (*DASH_GEMM_flt_ptr)(dash_cmplx_flt_type*, dash_cmplx_flt_type*, dash_cmplx_flt_type*, size_t, size_t, size_t);
static void (*DASH_FFT_flt_ptr)(dash_cmplx_flt_type*, dash_cmplx_flt_type*, size_t, bool);

void __attribute__((constructor)) setup_APIs_convolution(void) {

  void *handle;
  char *error;

  // Open the shared object file
  handle = dlopen("./libdash-rt/libdash-rt.so", RTLD_LAZY);
  if (!handle) {
      fprintf(stderr, "%s\n", dlerror());
      exit(0);
      //return 1;
  }

  // Load the function symbol
  dlerror();  // Clear any existing error
  // DASH_ZIP_flt_nb_ptr = (void (*)(dash_cmplx_flt_type*, dash_cmplx_flt_type*, dash_cmplx_flt_type*, size_t, zip_op_t, cedr_barrier_t*))dlsym(handle, "DASH_ZIP_flt_nb");
  DASH_CONV_2D_flt_nb_ptr = (void (*)(dash_re_flt_type*, int, int, dash_re_flt_type*, int, dash_re_flt_type*, cedr_barrier_t*))dlsym(handle, "DASH_CONV_2D_flt_nb");
  DASH_ZIP_flt_nb_ptr = (void (*)(dash_cmplx_flt_type*, dash_cmplx_flt_type*, dash_cmplx_flt_type*, size_t, zip_op_t))dlsym(handle, "DASH_ZIP_flt");
  // DASH_CONV_2D_flt_nb_ptr = (void (*)(dash_re_flt_type*, int, int, dash_re_flt_type*, int, dash_re_flt_type*))dlsym(handle, "DASH_CONV_2D_flt");
  DASH_GEMM_flt_ptr = (void (*)(dash_cmplx_flt_type*, dash_cmplx_flt_type*, dash_cmplx_flt_type*, size_t, size_t, size_t))dlsym(handle, "DASH_GEMM_flt");
  DASH_FFT_flt_ptr = (void (*)(dash_cmplx_flt_type*, dash_cmplx_flt_type*, size_t, bool))dlsym(handle, "DASH_FFT_flt");

  if ((error = dlerror()) != NULL) {
      fprintf(stderr, "%s\n", error);
      dlclose(handle);
      exit(0);
      //return 1;
  }

  dlclose(handle);

}

void cedr_conv2D(float *in, float  *filter, float *out, int height, int width, int kernel_size, int filter_number, int in_channel){
  dash_re_flt_type *conv_output = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * filter_number * in_channel * height * width);

  for(int i = 0; i < filter_number; i++){
    // Call non-blocking CONV_2D

    dash_re_flt_type **conv_in_0; // height x width
    dash_re_flt_type **conv_out_0; // height x width
    dash_re_flt_type **conv_filter_0; // kernel_size x kernel_size
    conv_in_0 = ((dash_re_flt_type**) calloc(in_channel, sizeof(dash_re_flt_type*)));
    conv_out_0 = ((dash_re_flt_type**) calloc(in_channel, sizeof(dash_re_flt_type*)));
    conv_filter_0 = ((dash_re_flt_type**) calloc(in_channel, sizeof(dash_re_flt_type*)));


    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    uint32_t completion_ctr = 0;
    uint32_t completion = in_channel;
    cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr, .completion = &completion};

    // Set Input and Output pointers
    for(int j = 0; j < in_channel; j++){
      conv_in_0[j] = (dash_re_flt_type *) &(in[j * height * width]);
      conv_out_0[j] = (dash_re_flt_type *) &(conv_output[j * height * width]);
      conv_filter_0[j] = (dash_re_flt_type *) &(filter[((i * in_channel + j) * kernel_size * kernel_size)]);
    }
    // Call non-blocking CONV_2D
    for(int j = 0; j < in_channel; j++){
      DASH_CONV_2D_flt_nb(&conv_in_0[j], &height, &width, &conv_filter_0[j], &kernel_size, &conv_out_0[j], &barrier);
    }
    //Wait for all CONV_2D's to complete
    pthread_mutex_lock(barrier.mutex);
    if (completion_ctr != in_channel) {
      pthread_cond_wait(barrier.cond, barrier.mutex);
    }
    pthread_mutex_unlock(barrier.mutex);
    free(conv_in_0);
    free(conv_out_0);
    free(conv_filter_0);

    for(int j = 0; j < in_channel; j++){
			DASH_ZIP_flt((dash_cmplx_flt_type *)&conv_output[j * height * width], (dash_cmplx_flt_type *)&(out[(i * height * width)]), (dash_cmplx_flt_type *)&(out[(i * height * width)]), height * width / 2, ZIP_ADD);
		}
  }

  
  fflush(stdout);
  free(conv_output);
}

void transpose_linear_weight(dash_re_flt_type *in, dash_re_flt_type *out, int in_channel, int out_channel){
  for(int row = 0; row < in_channel; row++){
    for(int col = 0; col < out_channel; col++){
			 out[col * in_channel + row] = in[row * out_channel + col];
    }
	 }
}

void _cedr_linear(float* in, float* filter, float* bias, float* out, int channel, int in_channel, int out_channel, bool bias_flag){
  dash_re_flt_type *filter_t = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * in_channel * out_channel);
  transpose_linear_weight(filter, filter_t, out_channel, in_channel);

  dash_re_flt_type* out_re = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * channel * out_channel * 2);
  dash_re_flt_type* input = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * channel * in_channel * 2);
  dash_re_flt_type *filter_in = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * in_channel * out_channel * 2);

  for(int i = 0; i < channel * in_channel; i++){
    input[i * 2] = in[i];
    input[i * 2 + 1] = 0;
  }

  for(int i = 0; i < in_channel * out_channel; i++){
    filter_in[i * 2] = filter_t[i];
    filter_in[i * 2 + 1] = 0;
  }

  DASH_GEMM_flt_ptr((dash_cmplx_flt_type *)input, (dash_cmplx_flt_type *)filter_in, (dash_cmplx_flt_type *)out_re, channel, in_channel, out_channel);

  for(int i = 0; i < channel * out_channel; i++){
    out[i] = out_re[i * 2];
  }

  if (bias_flag){
    for(int i = 0; i < channel; i++){
      for(int j = 0; j < out_channel; j++){
        out[i * out_channel + j] += bias[j];
      }
    }
  }

  free(filter_t);
  free(out_re);
  free(input);
  free(filter_in);
}
