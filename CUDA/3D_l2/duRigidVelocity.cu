#include <cmath>
#include "matvec.h"
#include "constants.h"
#include "struct.h"
#include "rotation.h"

__global__ void duRigidVelocityKernel(double *d_duRVMat, double *d_difIniMat,
                                      double *d_ctlCumMat, double *d_rgtMat, int rgdNdeNum)
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

		vector rgtVec;
		getVector(rgtVec, d_rgtMat, rgdNdeIdx, rgdNdeNum);

		double duAng1Val = dotProduct(rgtVec, rotVlc1Vec);
		double duAng2Val = dotProduct(rgtVec, rotVlc2Vec);
		double duAng3Val = dotProduct(rgtVec, rotVlc3Vec);

		d_duRVMat[                rgdNdeIdx] = duAng1Val;
		d_duRVMat[    rgdNdeNum + rgdNdeIdx] = duAng2Val;
		d_duRVMat[2 * rgdNdeNum + rgdNdeIdx] = duAng3Val;
		d_duRVMat[3 * rgdNdeNum + rgdNdeIdx] = rgtVec.x;
		d_duRVMat[4 * rgdNdeNum + rgdNdeIdx] = rgtVec.y;
		d_duRVMat[5 * rgdNdeNum + rgdNdeIdx] = rgtVec.z;
	}

	return;
}

void duRigidVelocity(double *d_duRVMat, double *d_ctlCumMat, double *d_rgtMat, fcndata &fcnObj)
{
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;

	int blkNum = (rgdNdeNum - 1) / BLKDIM + 1;
	duRigidVelocityKernel <<<blkNum, BLKDIM>>> (d_duRVMat, fcnObj.prm.d_difIniMat,
	                                            d_ctlCumMat, d_rgtMat, rgdNdeNum);

	return;
}
