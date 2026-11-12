#include "dash.h"

void cedr_conv2D(float  *in, float  *filter, float *out, int height, int width, int kernel_size, int filter_number, int in_channel);
void _cedr_linear(float* in, float* filter, float* bias, float* out, int channel, int in_channel, int out_channel, bool bias_flag);
