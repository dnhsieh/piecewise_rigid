#include <cmath>
#include "matvec.h"
#include "constants.h"
#include "struct.h"

__global__ void rigidDeformationKernel(double *d_rgdNdeMat,
                                       double *d_difIniMat, double *d_ctlCumMat, int rgdNdeNum)
{
	int rgdNdeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( rgdNdeIdx < rgdNdeNum )
	{
		vector difIniVec;
		getVector(difIniVec, d_difIniMat, rgdNdeIdx, rgdNdeNum);

		double angCumVal = d_ctlCumMat[rgdNdeIdx];
		vector linCumVec;
		getVector(linCumVec, d_ctlCumMat + rgdNdeNum, rgdNdeIdx, rgdNdeNum);
	
		double cosVal = cos(angCumVal);
		double sinVal = sin(angCumVal);

		vector rgdNdeVec;
		rgdNdeVec.x = linCumVec.x + cosVal * difIniVec.x - sinVal * difIniVec.y;
		rgdNdeVec.y = linCumVec.y + sinVal * difIniVec.x + cosVal * difIniVec.y;

		setVector(d_rgdNdeMat, rgdNdeVec, rgdNdeIdx, rgdNdeNum);
	}

	return;
}

void computeRigidDeformation(double *d_rgdNdeMat, double *d_ctlCumMat, fcndata &fcnObj)
{
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;
	
	int blkNum = (rgdNdeNum - 1) / BLKDIM + 1;
	rigidDeformationKernel <<<blkNum, BLKDIM>>> (d_rgdNdeMat, fcnObj.prm.d_difIniMat,
	                                             d_ctlCumMat, rgdNdeNum);

	return;
}
