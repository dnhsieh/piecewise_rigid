#include <cmath>
#include "matvec.h"
#include "constants.h"
#include "struct.h"
#include "rotation.h"

__global__ void dqRigidVelocityKernel(double *d_dqRVMat, double *d_difIniMat,
                                      double *d_ctlNdeMat, double *d_ctlCumMat, double *d_rgtMat, int rgdNdeNum)
{
	int rgdNdeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( rgdNdeIdx < rgdNdeNum )
	{
		vector difIniVec;
		getVector(difIniVec, d_difIniMat, rgdNdeIdx, rgdNdeNum);

		vector angVlcVec;
		getVector(angVlcVec, d_ctlNdeMat, rgdNdeIdx, rgdNdeNum);

		vector angCumVec;
		getVector(angCumVec, d_ctlCumMat, rgdNdeIdx, rgdNdeNum);
		
		// ---

		vector dqAng11Vec;	
		dqAng11Vec = applyD2RotXMat( difIniVec, angCumVec);
		dqAng11Vec =   applyRotYMat(dqAng11Vec, angCumVec);
		dqAng11Vec =   applyRotZMat(dqAng11Vec, angCumVec);

		vector dqAng12Vec;
		dqAng12Vec =  applyDRotXMat( difIniVec, angCumVec);
		dqAng12Vec =  applyDRotYMat(dqAng12Vec, angCumVec);
		dqAng12Vec =   applyRotZMat(dqAng12Vec, angCumVec);

		vector dqAng13Vec;
		dqAng13Vec =  applyDRotXMat( difIniVec, angCumVec);
		dqAng13Vec =   applyRotYMat(dqAng13Vec, angCumVec);
		dqAng13Vec =  applyDRotZMat(dqAng13Vec, angCumVec);

		vector dqAng1Vec;
		dqAng1Vec.x = angVlcVec.x * dqAng11Vec.x + angVlcVec.y * dqAng12Vec.x + angVlcVec.z * dqAng13Vec.x;
		dqAng1Vec.y = angVlcVec.x * dqAng11Vec.y + angVlcVec.y * dqAng12Vec.y + angVlcVec.z * dqAng13Vec.y;
		dqAng1Vec.z = angVlcVec.x * dqAng11Vec.z + angVlcVec.y * dqAng12Vec.z + angVlcVec.z * dqAng13Vec.z;

		// ---

		vector dqAng21Vec;	
		dqAng21Vec =  applyDRotXMat( difIniVec, angCumVec);
		dqAng21Vec =  applyDRotYMat(dqAng21Vec, angCumVec);
		dqAng21Vec =   applyRotZMat(dqAng21Vec, angCumVec);

		vector dqAng22Vec;
		dqAng22Vec =   applyRotXMat( difIniVec, angCumVec);
		dqAng22Vec = applyD2RotYMat(dqAng22Vec, angCumVec);
		dqAng22Vec =   applyRotZMat(dqAng22Vec, angCumVec);

		vector dqAng23Vec;
		dqAng23Vec =   applyRotXMat( difIniVec, angCumVec);
		dqAng23Vec =  applyDRotYMat(dqAng23Vec, angCumVec);
		dqAng23Vec =  applyDRotZMat(dqAng23Vec, angCumVec);

		vector dqAng2Vec;
		dqAng2Vec.x = angVlcVec.x * dqAng21Vec.x + angVlcVec.y * dqAng22Vec.x + angVlcVec.z * dqAng23Vec.x;
		dqAng2Vec.y = angVlcVec.x * dqAng21Vec.y + angVlcVec.y * dqAng22Vec.y + angVlcVec.z * dqAng23Vec.y;
		dqAng2Vec.z = angVlcVec.x * dqAng21Vec.z + angVlcVec.y * dqAng22Vec.z + angVlcVec.z * dqAng23Vec.z;

		// ---

		vector dqAng31Vec;	
		dqAng31Vec =  applyDRotXMat( difIniVec, angCumVec);
		dqAng31Vec =   applyRotYMat(dqAng31Vec, angCumVec);
		dqAng31Vec =  applyDRotZMat(dqAng31Vec, angCumVec);

		vector dqAng32Vec;
		dqAng32Vec =   applyRotXMat( difIniVec, angCumVec);
		dqAng32Vec =  applyDRotYMat(dqAng32Vec, angCumVec);
		dqAng32Vec =  applyDRotZMat(dqAng32Vec, angCumVec);

		vector dqAng33Vec;
		dqAng33Vec =   applyRotXMat( difIniVec, angCumVec);
		dqAng33Vec =   applyRotYMat(dqAng33Vec, angCumVec);
		dqAng33Vec = applyD2RotZMat(dqAng33Vec, angCumVec);

		vector dqAng3Vec;
		dqAng3Vec.x = angVlcVec.x * dqAng31Vec.x + angVlcVec.y * dqAng32Vec.x + angVlcVec.z * dqAng33Vec.x;
		dqAng3Vec.y = angVlcVec.x * dqAng31Vec.y + angVlcVec.y * dqAng32Vec.y + angVlcVec.z * dqAng33Vec.y;
		dqAng3Vec.z = angVlcVec.x * dqAng31Vec.z + angVlcVec.y * dqAng32Vec.z + angVlcVec.z * dqAng33Vec.z;

		// ---

		vector rgtVec;
		getVector(rgtVec, d_rgtMat, rgdNdeIdx, rgdNdeNum);

		double dqAng1Val = dotProduct(rgtVec, dqAng1Vec);
		double dqAng2Val = dotProduct(rgtVec, dqAng2Vec);
		double dqAng3Val = dotProduct(rgtVec, dqAng3Vec);

		d_dqRVMat[                rgdNdeIdx] = dqAng1Val;
		d_dqRVMat[    rgdNdeNum + rgdNdeIdx] = dqAng2Val;
		d_dqRVMat[2 * rgdNdeNum + rgdNdeIdx] = dqAng3Val;
		d_dqRVMat[3 * rgdNdeNum + rgdNdeIdx] = 0.0;
		d_dqRVMat[4 * rgdNdeNum + rgdNdeIdx] = 0.0;
		d_dqRVMat[5 * rgdNdeNum + rgdNdeIdx] = 0.0;
	}

	return;
}

void dqRigidVelocity(double *d_dqRVMat, double *d_ctlNdeMat, double *d_ctlCumMat,
                     double *d_rgtMat, fcndata &fcnObj)
{
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;

	int blkNum = (rgdNdeNum - 1) / BLKDIM + 1;
	dqRigidVelocityKernel <<<blkNum, BLKDIM>>> (d_dqRVMat, fcnObj.prm.d_difIniMat,
	                                            d_ctlNdeMat, d_ctlCumMat, d_rgtMat, rgdNdeNum);

	return;
}
