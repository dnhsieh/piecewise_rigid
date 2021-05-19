#include "constants.h"

__global__ void addEpsKernel(double *d_squMat, double epsVal, int rowNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( rowIdx < rowNum )
	{
		int glbIdx = rowIdx * rowNum + rowIdx;
		d_squMat[glbIdx] += epsVal;
	}

	return;
}

void addEpsIdentity(double *d_squMat, double epsVal, int rowNum)
{
	int blkNum = (rowNum - 1) / BLKDIM + 1;
	addEpsKernel <<<blkNum, BLKDIM>>> (d_squMat, epsVal, rowNum);

	return;
}
