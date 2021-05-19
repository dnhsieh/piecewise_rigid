#include <cmath>
#include "matvec.h"
#include "constants.h"
#include "struct.h"

__global__ void rigidVelocityKernel(double *d_rgdVlcMat, double *d_difIniMat,
                                    double *d_ctlNdeMat, double *d_ctlCumMat, int rgdNdeNum)
{
	int rgdNdeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( rgdNdeIdx < rgdNdeNum )
	{
		double angCumVal = d_ctlCumMat[rgdNdeIdx];
		
		double cosVal = cos(angCumVal);
		double sinVal = sin(angCumVal);

		double angVlcVal = d_ctlNdeMat[rgdNdeIdx];

		vector linVlcVec;
		getVector(linVlcVec, d_ctlNdeMat + rgdNdeNum, rgdNdeIdx, rgdNdeNum);

		vector difIniVec;
		getVector(difIniVec, d_difIniMat, rgdNdeIdx, rgdNdeNum);

		vector rgdVlcVec;
		rgdVlcVec.x = linVlcVec.x + angVlcVal * (-sinVal * difIniVec.x - cosVal * difIniVec.y);
		rgdVlcVec.y = linVlcVec.y + angVlcVal * ( cosVal * difIniVec.x - sinVal * difIniVec.y);

		setVector(d_rgdVlcMat, rgdVlcVec, rgdNdeIdx, rgdNdeNum);
	}

	return;
}

void computeRigidVelocity(double *d_rgdVlcMat, double *d_ctlNdeMat, double *d_ctlCumMat, fcndata &fcnObj)
{
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;

	int blkNum = (rgdNdeNum - 1) / BLKDIM + 1;
	rigidVelocityKernel <<<blkNum, BLKDIM>>> (d_rgdVlcMat, fcnObj.prm.d_difIniMat,
	                                          d_ctlNdeMat, d_ctlCumMat, rgdNdeNum);

	return;
}
