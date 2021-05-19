#include "constants.h"

__global__ void scaleKernel(double *d_out, double sVal, double *d_inp, int len)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < len )
		d_out[idx] = sVal * d_inp[idx];

	return;
}

void vectorScale(double *d_out, double sVal, double *d_inp, int len)
{
	int blkNum = (len - 1) / BLKDIM + 1;
	scaleKernel <<<blkNum, BLKDIM>>> (d_out, sVal, d_inp, len);

	return;
}
