#include "constants.h"
#include "struct.h"

__global__ void duControlKernel(double *d_grdGrpStk, double *d_ctlGrpStk, double *d_wgtGrpVec,
                                int grpNum, int ttlNum)
{
	int ttlIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( ttlIdx < ttlNum )
	{
		int timeIdx = ttlIdx / grpNum;
		int  grpIdx = ttlIdx - timeIdx * grpNum;

		double wgtVal = d_wgtGrpVec[grpIdx];
		for ( int dofIdx = 0; dofIdx < RGDDOF; ++dofIdx )
		{
			d_grdGrpStk[(timeIdx * RGDDOF + dofIdx) * grpNum + grpIdx] +=
			   wgtVal * d_ctlGrpStk[(timeIdx * RGDDOF + dofIdx) * grpNum + grpIdx];
		}
	}

	return;
}

void duControl(double *d_grdGrpStk, double *d_ctlGrpStk, double *d_wgtGrpVec, fcndata &fcnObj)
{
	int rgdGrpNum = fcnObj.prm.rgdGrpNum;
	int   timeNum = fcnObj.prm.timeNum;

	int ttlNum = rgdGrpNum * (timeNum - 1);
	int blkNum = (ttlNum - 1) / BLKDIM + 1;
	duControlKernel <<<blkNum, BLKDIM>>> (d_grdGrpStk, d_ctlGrpStk, d_wgtGrpVec, rgdGrpNum, ttlNum);

	return;
}
