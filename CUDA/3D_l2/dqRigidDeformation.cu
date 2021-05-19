#include <cmath>
#include "matvec.h"
#include "constants.h"
#include "struct.h"
#include "rotation.h"

__global__ void dqRigidDeformationKernel(double *d_dqRVMat, double *d_difIniMat,
                                         double *d_ctlCumMat, double *d_rgtMat, int rgdNdeNum)
{
	int rgdNdeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( rgdNdeIdx < rgdNdeNum )
	{
		vector difIniVec;
		getVector(difIniVec, d_difIniMat, rgdNdeIdx, rgdNdeNum);

		vector angCumVec;
		getVector(angCumVec, d_ctlCumMat, rgdNdeIdx, rgdNdeNum);
	
		vector dqAng1Vec;
		dqAng1Vec = applyDRotXMat(difIniVec, angCumVec);
		dqAng1Vec =  applyRotYMat(dqAng1Vec, angCumVec);
		dqAng1Vec =  applyRotZMat(dqAng1Vec, angCumVec);

		vector dqAng2Vec;
		dqAng2Vec =  applyRotXMat(difIniVec, angCumVec);
		dqAng2Vec = applyDRotYMat(dqAng2Vec, angCumVec);
		dqAng2Vec =  applyRotZMat(dqAng2Vec, angCumVec);

		vector dqAng3Vec;
		dqAng3Vec =  applyRotXMat(difIniVec, angCumVec);
		dqAng3Vec =  applyRotYMat(dqAng3Vec, angCumVec);
		dqAng3Vec = applyDRotZMat(dqAng3Vec, angCumVec);

		// ---

		vector rgtVec;
		getVector(rgtVec, d_rgtMat, rgdNdeIdx, rgdNdeNum);

		double dqAng1Val = dotProduct(rgtVec, dqAng1Vec);
		double dqAng2Val = dotProduct(rgtVec, dqAng2Vec);
		double dqAng3Val = dotProduct(rgtVec, dqAng3Vec);

		d_dqRVMat[                rgdNdeIdx] = dqAng1Val;
		d_dqRVMat[    rgdNdeNum + rgdNdeIdx] = dqAng2Val;
		d_dqRVMat[2 * rgdNdeNum + rgdNdeIdx] = dqAng3Val;
		d_dqRVMat[3 * rgdNdeNum + rgdNdeIdx] = rgtVec.x;
		d_dqRVMat[4 * rgdNdeNum + rgdNdeIdx] = rgtVec.y;
		d_dqRVMat[5 * rgdNdeNum + rgdNdeIdx] = rgtVec.z;
	}

	return;
}

void dqRigidDeformation(double *d_dqRVMat, double *d_ctlCumMat, double *d_rgtMat, fcndata &fcnObj)
{
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;

	int blkNum = (rgdNdeNum - 1) / BLKDIM + 1;
	dqRigidDeformationKernel <<<blkNum, BLKDIM>>> (d_dqRVMat, fcnObj.prm.d_difIniMat,
	                                               d_ctlCumMat, d_rgtMat, rgdNdeNum);

	return;
}
