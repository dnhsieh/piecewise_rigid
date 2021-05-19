#include <cmath>
#include "matvec.h"
#include "constants.h"
#include "struct.h"

__global__ void dqRigidVelocityKernel(double *d_dqRVMat, double *d_difIniMat,
                                      double *d_ctlNdeMat, double *d_ctlCumMat, double *d_lftMat, int rgdNdeNum)
{
	int rgdNdeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( rgdNdeIdx < rgdNdeNum )
	{
		double angVlcVal = d_ctlNdeMat[rgdNdeIdx];
		double angCumVal = d_ctlCumMat[rgdNdeIdx];

		double cosVal = cos(angCumVal);
		double sinVal = sin(angCumVal);

		vector difIniVec, lftVec;
		getVector(difIniVec, d_difIniMat, rgdNdeIdx, rgdNdeNum);
		getVector(lftVec, d_lftMat, rgdNdeIdx, rgdNdeNum);

		double dqAngVal = angVlcVal * (  lftVec.x * (-cosVal * difIniVec.x + sinVal * difIniVec.y)
		                               + lftVec.y * (-sinVal * difIniVec.x - cosVal * difIniVec.y) );

		d_dqRVMat[                rgdNdeIdx] = dqAngVal;
		d_dqRVMat[    rgdNdeNum + rgdNdeIdx] = 0.0;
		d_dqRVMat[2 * rgdNdeNum + rgdNdeIdx] = 0.0;
	}

	return;
}

void dqRigidVelocity(double *d_dqRVMat, double *d_ctlNdeMat, double *d_ctlCumMat,
                     double *d_lftMat, fcndata &fcnObj)
{
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;

	int blkNum = (rgdNdeNum - 1) / BLKDIM + 1;
	dqRigidVelocityKernel <<<blkNum, BLKDIM>>> (d_dqRVMat, fcnObj.prm.d_difIniMat,
	                                            d_ctlNdeMat, d_ctlCumMat, d_lftMat, rgdNdeNum);

	return;
}
