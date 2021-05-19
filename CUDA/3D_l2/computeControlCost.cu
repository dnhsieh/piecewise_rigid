#include "constants.h"
#include "struct.h"

void dsum(double *, double *, double *, int);

__global__ void controlCostKernel(double *d_ctlVec, double *d_ctlGrpStk,
                                  double *d_wgtGrpVec, int grpNum, int ttlNum)
{
	int ttlIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( ttlIdx < ttlNum )
	{
		int    grpIdx = ttlIdx % grpNum;
		double wgtVal = d_wgtGrpVec[grpIdx];
		
		double ctlVal = 0.0;
		for ( int dofIdx = 0; dofIdx < RGDDOF; ++dofIdx )
		{
			ctlVal += wgtVal * d_ctlGrpStk[dofIdx * ttlNum + ttlIdx]
			                 * d_ctlGrpStk[dofIdx * ttlNum + ttlIdx];
		}

		d_ctlVec[ttlIdx] = ctlVal;
	}

	return;
}

double computeControlCost(double *d_ctlGrpStk, double *d_wgtGrpVec, fcndata &fcnObj)
{
	int       grpNum = fcnObj.prm.rgdGrpNum;
	int      timeNum = fcnObj.prm.timeNum;
	double *d_ctlVec = fcnObj.d_ctlVec;

	int ttlNum = grpNum * (timeNum - 1);
	int blkNum = ttlNum / BLKDIM + 1;
	controlCostKernel <<<blkNum, BLKDIM>>> (d_ctlVec, d_ctlGrpStk, d_wgtGrpVec, grpNum, ttlNum);

	double h_cstVal;
	dsum(&h_cstVal, d_ctlVec, fcnObj.d_sumBufVec, grpNum * (timeNum - 1));

	return 0.5 * h_cstVal;
}
