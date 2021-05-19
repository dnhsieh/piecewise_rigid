#include "constants.h"
#include "struct.h"

__global__ void group2NodeKernel(double *d_ndeStk, double *d_grpStk, int *d_grpNdeVec,
                                 int ndeNum, int grpNum, int ttlNum)
{
	int ttlIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( ttlIdx < ttlNum )
	{
		int timeIdx = ttlIdx / (ndeNum * RGDDOF);
		int dofIdx  = (ttlIdx - timeIdx * ndeNum * RGDDOF) / ndeNum;
		int ndeIdx  = ttlIdx - (timeIdx * RGDDOF + dofIdx) * ndeNum;

		int grpIdx = d_grpNdeVec[ndeIdx];
		d_ndeStk[(timeIdx * RGDDOF + dofIdx) * ndeNum + ndeIdx] = 
		   d_grpStk[(timeIdx * RGDDOF + dofIdx) * grpNum + grpIdx];
	}

	return;
}

void group2Node(double *d_ndeStk, double *d_grpStk, int *d_grpNdeVec,
                int ndeNum, int grpNum, int timeNum)
{
	int ttlNum = ndeNum * RGDDOF * (timeNum - 1);
	int blkNum = (ttlNum - 1) / BLKDIM + 1;
	group2NodeKernel <<<blkNum, BLKDIM>>> (d_ndeStk, d_grpStk, d_grpNdeVec, ndeNum, grpNum, ttlNum);

	return;
}
