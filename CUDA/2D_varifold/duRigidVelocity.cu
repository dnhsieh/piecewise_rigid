#include <cmath>
#include "matvec.h"
#include "constants.h"
#include "struct.h"

__global__ void duRigidVelocityKernel(double *d_duRVMat, double *d_difIniMat,
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

		double duAngVal =  lftVec.x * (-sinVal * difIniVec.x - cosVal * difIniVec.y)
                       + lftVec.y * ( cosVal * difIniVec.x - sinVal * difIniVec.y);

		d_duRVMat[                rgdNdeIdx] = duAngVal;
		d_duRVMat[    rgdNdeNum + rgdNdeIdx] = lftVec.x;
		d_duRVMat[2 * rgdNdeNum + rgdNdeIdx] = lftVec.y;
	}

	return;
}

void duRigidVelocity(double *d_duRVMat, double *d_ctlCumMat, double *d_lftMat, fcndata &fcnObj)
{
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;

	int blkNum = (rgdNdeNum - 1) / BLKDIM + 1;
	duRigidVelocityKernel <<<blkNum, BLKDIM>>> (d_duRVMat, fcnObj.prm.d_difIniMat,
	                                            d_ctlCumMat, d_lftMat, rgdNdeNum);

	return;
}
