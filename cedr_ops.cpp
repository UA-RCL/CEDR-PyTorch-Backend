#include <iostream>
#include <omp.h>
#include <dlfcn.h>
#include <cmath>

#include "include/dash.h"

extern void enqueue_kernel(const char* kernel_name, const char* precision_name, unsigned int n_vargs, ...);

// static void (*DASH_ZIP_flt_nb_ptr)(dash_cmplx_flt_type*, dash_cmplx_flt_type*, dash_cmplx_flt_type*, size_t, zip_op_t, cedr_barrier_t*);
static void (*DASH_CONV_2D_flt_nb_ptr)(dash_re_flt_type*, int, int, dash_re_flt_type*, int, dash_re_flt_type*, cedr_barrier_t*);
static void (*DASH_ZIP_flt_nb_ptr)(dash_cmplx_flt_type*, dash_cmplx_flt_type*, dash_cmplx_flt_type*, size_t, zip_op_t);
// static void (*DASH_CONV_2D_flt_nb_ptr)(dash_re_flt_type*, int, int, dash_re_flt_type*, int, dash_re_flt_type*);
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

    // pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    // pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    // uint32_t completion_ctr = 0;
    // uint32_t completion = in_channel;
    // cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr, .completion = &completion};

    // for(int j = 0; j < in_channel; j++){
    //   DASH_CONV_2D_flt_nb_ptr(&in[j * height * width], height, width, &filter[((i * in_channel + j) * kernel_size * kernel_size)], kernel_size, &conv_output[i*in_channel*height*width + j * height * width], &barrier);
    //   // DASH_ZIP_flt_nb_ptr((dash_cmplx_flt_type *)&conv_output[i *  in_channel * height * width + j * height * width], (dash_cmplx_flt_type *)&(out[(i * height * width)]), (dash_cmplx_flt_type *)&(out[(i * height * width)]), height * width / 2, ZIP_ADD, &barrier);
    //   // DASH_CONV_2D_flt_nb_ptr(&in[j * height * width], height, width, &filter[((i * in_channel + j) * kernel_size * kernel_size)], kernel_size, &conv_output[i*in_channel*height*width + j * height * width]);
    //   // DASH_ZIP_flt_nb_ptr((dash_cmplx_flt_type *)&conv_output[i *  in_channel * height * width + j * height * width], (dash_cmplx_flt_type *)&(out[(i * height * width)]), (dash_cmplx_flt_type *)&(out[(i * height * width)]), height * width / 2, ZIP_ADD);
    // }

    // pthread_mutex_lock(barrier.mutex);
    // if (completion_ctr != in_channel) {
    //   pthread_cond_wait(barrier.cond, barrier.mutex);
    // }
    // pthread_mutex_unlock(barrier.mutex);

    // for(int j = 0; j < in_channel; j++){
    //   // DASH_ZIP_flt_nb_ptr((dash_cmplx_flt_type *)&conv_output[i *  in_channel * height * width + j * height * width], (dash_cmplx_flt_type *)&(out[(i * height * width)]), (dash_cmplx_flt_type *)&(out[(i * height * width)]), height * width / 2, ZIP_ADD, &barrier);
    //   DASH_ZIP_flt_nb_ptr((dash_cmplx_flt_type *)&conv_output[i *  in_channel * height * width + j * height * width], (dash_cmplx_flt_type *)&(out[(i * height * width)]), (dash_cmplx_flt_type *)&(out[(i * height * width)]), height * width / 2, ZIP_ADD);
    // }
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

void _cedr_conv1D_FFT_pad(dash_re_flt_type *in, dash_re_flt_type *out, int width, int padded_width){    
  for(int j = 0; j < padded_width; j++){
      out[j] = 0;
      if(j < width){
          out[j] = in[j];
      }
  }
}

void _cedr_conv1D_FFT_complex(dash_re_flt_type *in, dash_re_flt_type *out, int width){
  for(int col = 0; col < width; col++){
    out[col * 2] = in[col];
    out[col * 2 + 1] = 0;
  }
}

void _cedr_conv1D_FFT(float *in, float  *filter, float *out, int width, int kernel_size, int filter_number, int in_channel){
  for(int filter_iter = 0; filter_iter < filter_number; filter_iter++){
    dash_re_flt_type *filter_group = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (in_channel * kernel_size));

    int idx = 0;
    for(int i = filter_iter * (in_channel * kernel_size); i < (filter_iter + 1) * (in_channel * kernel_size); i++){
      filter_group[idx++] = filter[i];
    }

    for(int channel_iter = 0; channel_iter < in_channel; channel_iter++){

      dash_re_flt_type *temp_filter = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (kernel_size));
      dash_re_flt_type *temp_input = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (width));

      for(int i = 0; i < kernel_size; i++){
        temp_filter[i] = filter_group[channel_iter * kernel_size + i];
      }

      for(int i = 0; i < width; i++){
        temp_input[i] = in[channel_iter * width + i];
      }

      int pad = kernel_size / 2;
      int kpad_width = width + (pad * 2);
      dash_re_flt_type *kpad_input = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (kpad_width));

      for(int j = 0; j < kpad_width; j++){
        // if(j < pad || j >= width + pad)kpad_input[j] = 0;
        // else kpad_input[j] = temp_input[j - pad];
        if(j >= width + pad * 2)kpad_input[j] = 0;
        else kpad_input[j] = temp_input[j];
      } 

      int pad_w = pow(2, ceil(log2(kpad_width)));
      dash_re_flt_type *padded_filter = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (pad_w));
      dash_re_flt_type *padded_input = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (pad_w));
      _cedr_conv1D_FFT_pad(filter, padded_filter, kernel_size, pad_w);
      _cedr_conv1D_FFT_pad(kpad_input, padded_input, kpad_width, pad_w);

      int complex_w = pad_w * 2;
      dash_re_flt_type *complex_filter = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (complex_w));
      dash_re_flt_type *complex_input = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (complex_w));
      _cedr_conv1D_FFT_complex(padded_filter, complex_filter, pad_w);
      _cedr_conv1D_FFT_complex(padded_input, complex_input, pad_w);

      dash_cmplx_flt_type *fft_filter = (dash_cmplx_flt_type *) malloc (sizeof (dash_cmplx_flt_type) * (complex_w));
      dash_cmplx_flt_type *fft_input = (dash_cmplx_flt_type *) malloc (sizeof (dash_cmplx_flt_type) * (complex_w));

      DASH_FFT_flt_ptr((dash_cmplx_flt_type*)complex_filter, fft_filter, pad_w, true);
      DASH_FFT_flt_ptr((dash_cmplx_flt_type*)complex_input, fft_input, pad_w, true);

      for(int i = 0; i < complex_w/2; i++){
        fft_filter[i].im = -fft_filter[i].im;
      }

      DASH_ZIP_flt_nb_ptr(fft_input, fft_filter, fft_input, complex_w / 2, ZIP_MULT);

      dash_cmplx_flt_type *ifft_output = (dash_cmplx_flt_type *) malloc (sizeof (dash_cmplx_flt_type) * (complex_w));
      DASH_FFT_flt_ptr((dash_cmplx_flt_type*)fft_input, ifft_output, pad_w, false);

      int cropped_size = width - kernel_size + 3;
      for(int j = 0; j < cropped_size; j++){
          out[filter_iter * cropped_size + j] += (dash_re_flt_type)ifft_output[j].re;
      }

      free(temp_filter);
      free(temp_input);
      free(kpad_input);
      free(padded_filter);
      free(padded_input);
      free(complex_filter);
      free(complex_input);
      free(fft_filter);
      free(fft_input);
      free(ifft_output);

    }

    free(filter_group);

  }


}