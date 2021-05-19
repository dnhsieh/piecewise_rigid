#include "constants.h"

__global__ void sum2Kernel(double *d_out, double c1Val, double *d_inp1, double c2Val, double *d_inp2, int len)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < len )
		d_out[idx] = c1Val * d_inp1[idx] + c2Val * d_inp2[idx];

	return;
}

__global__ void sum3Kernel(double *d_out, double c1Val, double *d_inp1, double c2Val, double *d_inp2,
                                          double c3Val, double *d_inp3, int len)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < len )
	{
		d_out[idx] =  c1Val * d_inp1[idx] + c2Val * d_inp2[idx]
		            + c3Val * d_inp3[idx];
	}

	return;
}

__global__ void sum4Kernel(double *d_out, double c1Val, double *d_inp1, double c2Val, double *d_inp2,
                                          double c3Val, double *d_inp3, double c4Val, double *d_inp4, int len)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < len )
	{
		d_out[idx] =  c1Val * d_inp1[idx] + c2Val * d_inp2[idx]
		            + c3Val * d_inp3[idx] + c4Val * d_inp4[idx];
	}

	return;
}

void vectorSum(double *d_out, double c1Val, double *d_inp1, double c2Val, double *d_inp2, int len)
{
	int blkNum = (len - 1) / BLKDIM + 1;
	sum2Kernel <<<blkNum, BLKDIM>>> (d_out, c1Val, d_inp1, c2Val, d_inp2, len);

	return;
}

void vectorSum(double *d_out, double c1Val, double *d_inp1, double c2Val, double *d_inp2,
                              double c3Val, double *d_inp3, int len)
{
	int blkNum = (len - 1) / BLKDIM + 1;
	sum3Kernel <<<blkNum, BLKDIM>>> (d_out, c1Val, d_inp1, c2Val, d_inp2, c3Val, d_inp3, len);

	return;
}

void vectorSum(double *d_out, double c1Val, double *d_inp1, double c2Val, double *d_inp2,
                              double c3Val, double *d_inp3, double c4Val, double *d_inp4, int len)
{
	int blkNum = (len - 1) / BLKDIM + 1;
	sum4Kernel <<<blkNum, BLKDIM>>> (d_out, c1Val, d_inp1, c2Val, d_inp2, c3Val, d_inp3, c4Val, d_inp4, len);

	return;
}
