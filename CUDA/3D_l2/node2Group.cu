#include <cstdio>
#include "constants.h"

__global__ void node2GroupKernel(double *d_grpStk, double *d_ndeStk, int *d_grpIfoMat,
                                 int ndeNum, int grpNum, int ttlNum)
{
	int ttlIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( ttlIdx < ttlNum )
	{
		int timeIdx = ttlIdx / grpNum;
		int  grpIdx = ttlIdx - timeIdx * grpNum;

		double grpVec[RGDDOF] = {0.0};

		int mbrNum = d_grpIfoMat[grpIdx];
		for ( int mbrIdx = 0; mbrIdx < mbrNum; ++mbrIdx )
		{
			int ndeIdx = d_grpIfoMat[(1 + mbrIdx) * grpNum + grpIdx];
			for ( int dofIdx = 0; dofIdx < RGDDOF; ++dofIdx )
				grpVec[dofIdx] += d_ndeStk[(timeIdx * RGDDOF + dofIdx) * ndeNum + ndeIdx];
		}

		d_grpStk[(timeIdx * RGDDOF    ) * grpNum + grpIdx] = grpVec[0];
		d_grpStk[(timeIdx * RGDDOF + 1) * grpNum + grpIdx] = grpVec[1];
		d_grpStk[(timeIdx * RGDDOF + 2) * grpNum + grpIdx] = grpVec[2];
		d_grpStk[(timeIdx * RGDDOF + 3) * grpNum + grpIdx] = grpVec[3];
		d_grpStk[(timeIdx * RGDDOF + 4) * grpNum + grpIdx] = grpVec[4];
		d_grpStk[(timeIdx * RGDDOF + 5) * grpNum + grpIdx] = grpVec[5];
	}

	return;
}

void node2Group(double *d_grpStk, double *d_ndeStk, int *d_grpIfoMat, int ndeNum, int grpNum, int timeNum)
{
	int ttlNum = grpNum * (timeNum - 1);
	int blkNum = (ttlNum - 1) / BLKDIM + 1;
	node2GroupKernel <<<blkNum, BLKDIM>>> (d_grpStk, d_ndeStk, d_grpIfoMat, ndeNum, grpNum, ttlNum);

	return;
}
