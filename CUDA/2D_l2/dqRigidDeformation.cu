#include <cmath>
#include "matvec.h"
#include "constants.h"
#include "struct.h"

__global__ void dqRigidDeformationKernel(double *d_dqRVMat, double *d_difIniMat,
                                         double *d_ctlCumMat, double *d_lftMat, int rgdNdeNum)
{
	int rgdNdeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( rgdNdeIdx < rgdNdeNum )
	{
		double angCumVal = d_ctlCumMat[rgdNdeIdx];

		double cosVal = cos(angCumVal);
		double sinVal = sin(angCumVal);

		vector difIniVec, lftVec;
		getVector(difIniVec, d_difIniMat, rgdNdeIdx, rgdNdeNum);
		getVector(lftVec, d_lftMat, rgdNdeIdx, rgdNdeNum);

		double dqAngVal =  lftVec.x * (-sinVal * difIniVec.x - cosVal * difIniVec.y)
		                 + lftVec.y * ( cosVal * difIniVec.x - sinVal * difIniVec.y);

		d_dqRVMat[                rgdNdeIdx] = dqAngVal;
		d_dqRVMat[    rgdNdeNum + rgdNdeIdx] = lftVec.x;
		d_dqRVMat[2 * rgdNdeNum + rgdNdeIdx] = lftVec.y;
	}

	return;
}

void dqRigidDeformation(double *d_dqRVMat, double *d_ctlCumMat, double *d_lftMat, fcndata &fcnObj)
{
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;

	int blkNum = (rgdNdeNum - 1) / BLKDIM + 1;
	dqRigidDeformationKernel <<<blkNum, BLKDIM>>> (d_dqRVMat, fcnObj.prm.d_difIniMat,
	                                               d_ctlCumMat, d_lftMat, rgdNdeNum);

	return;
}
