#include <cmath>
#include "matvec.h"
#include "constants.h"
#include "struct.h"
#include "rotation.h"

__global__ void rigidDeformationKernel(double *d_rgdNdeMat,
                                       double *d_difIniMat, double *d_ctlCumMat, int rgdNdeNum)
{
	int rgdNdeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( rgdNdeIdx < rgdNdeNum )
	{
		vector difIniVec;
		getVector(difIniVec, d_difIniMat, rgdNdeIdx, rgdNdeNum);

		vector angCumVec, linCumVec;
		getVector(angCumVec, d_ctlCumMat,                      rgdNdeIdx, rgdNdeNum);
		getVector(linCumVec, d_ctlCumMat + DIMNUM * rgdNdeNum, rgdNdeIdx, rgdNdeNum);
	
		vector rotNdeVec;
		rotNdeVec = applyRotXMat(difIniVec, angCumVec);
		rotNdeVec = applyRotYMat(rotNdeVec, angCumVec);
		rotNdeVec = applyRotZMat(rotNdeVec, angCumVec);

		vector rgdNdeVec;
		vectorSum(rgdNdeVec, linCumVec, rotNdeVec);

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
