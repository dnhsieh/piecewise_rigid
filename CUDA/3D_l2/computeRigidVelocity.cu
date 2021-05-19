#include <cmath>
#include "matvec.h"
#include "constants.h"
#include "struct.h"
#include "rotation.h"

__global__ void rigidVelocityKernel(double *d_rgdVlcMat, double *d_difIniMat,
                                    double *d_ctlNdeMat, double *d_ctlCumMat, int rgdNdeNum)
{
	int rgdNdeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( rgdNdeIdx < rgdNdeNum )
	{
		vector difIniVec;
		getVector(difIniVec, d_difIniMat, rgdNdeIdx, rgdNdeNum);

		vector angCumVec;
		getVector(angCumVec, d_ctlCumMat, rgdNdeIdx, rgdNdeNum);
		
		vector rotVlc1Vec;
		rotVlc1Vec = applyDRotXMat( difIniVec, angCumVec);
		rotVlc1Vec =  applyRotYMat(rotVlc1Vec, angCumVec);
		rotVlc1Vec =  applyRotZMat(rotVlc1Vec, angCumVec);

		vector rotVlc2Vec;
		rotVlc2Vec =  applyRotXMat( difIniVec, angCumVec);
		rotVlc2Vec = applyDRotYMat(rotVlc2Vec, angCumVec);
		rotVlc2Vec =  applyRotZMat(rotVlc2Vec, angCumVec);

		vector rotVlc3Vec;
		rotVlc3Vec =  applyRotXMat( difIniVec, angCumVec);
		rotVlc3Vec =  applyRotYMat(rotVlc3Vec, angCumVec);
		rotVlc3Vec = applyDRotZMat(rotVlc3Vec, angCumVec);

		// ---

		vector angVlcVec, linVlcVec;
		getVector(angVlcVec, d_ctlNdeMat,                      rgdNdeIdx, rgdNdeNum);
		getVector(linVlcVec, d_ctlNdeMat + DIMNUM * rgdNdeNum, rgdNdeIdx, rgdNdeNum);

		vector rgdVlcVec;
		rgdVlcVec.x = linVlcVec.x + angVlcVec.x * rotVlc1Vec.x + angVlcVec.y * rotVlc2Vec.x + angVlcVec.z * rotVlc3Vec.x;
		rgdVlcVec.y = linVlcVec.y + angVlcVec.x * rotVlc1Vec.y + angVlcVec.y * rotVlc2Vec.y + angVlcVec.z * rotVlc3Vec.y;
		rgdVlcVec.z = linVlcVec.z + angVlcVec.x * rotVlc1Vec.z + angVlcVec.y * rotVlc2Vec.z + angVlcVec.z * rotVlc3Vec.z;

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
