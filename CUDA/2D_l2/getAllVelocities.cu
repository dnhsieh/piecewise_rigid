#include "constants.h"
#include "struct.h"

__global__ void getRigidVelocitiesKernel(double *d_allVlcMat, double *d_adjVlcMat,
                                         int allNdeNum, int rgdNdeNum, int ttlNum)
{
	int ttlIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( ttlIdx < ttlNum )
	{
		int    dimIdx = ttlIdx / rgdNdeNum;
		int rgdNdeIdx = ttlIdx - dimIdx * rgdNdeNum;
		
		d_allVlcMat[dimIdx * allNdeNum + rgdNdeIdx] = d_adjVlcMat[dimIdx * rgdNdeNum + rgdNdeIdx];
	}

	return;
}

__global__ void getElasticVelocitiesKernel(double *d_allVlcMat, double *d_elaVlcMat,
                                           int allNdeNum, int rgdNdeNum, int elaNdeNum, int ttlNum)
{
	int ttlIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( ttlIdx < ttlNum )
	{
		int    dimIdx = ttlIdx / elaNdeNum;
		int elaNdeIdx = ttlIdx - dimIdx * elaNdeNum;
		
		d_allVlcMat[dimIdx * allNdeNum + rgdNdeNum + elaNdeIdx] = d_elaVlcMat[dimIdx * elaNdeNum + elaNdeIdx];
	}

	return;
}

void getAllVelocities(double *d_allVlcMat, double *d_adjVlcMat, double *d_elaVlcMat, fcndata &fcnObj)
{
	int allNdeNum = fcnObj.prm.allNdeNum;
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;
	int elaNdeNum = fcnObj.prm.elaNdeNum;

	int ttlNum = rgdNdeNum * DIMNUM;
	int blkNum = (ttlNum - 1) / BLKDIM + 1;
	getRigidVelocitiesKernel <<<blkNum, BLKDIM>>> (d_allVlcMat, d_adjVlcMat, allNdeNum, rgdNdeNum, ttlNum);

	ttlNum = elaNdeNum * DIMNUM;
	blkNum = (ttlNum - 1) / BLKDIM + 1;
	getElasticVelocitiesKernel <<<blkNum, BLKDIM>>> (d_allVlcMat, d_elaVlcMat,
	                                                 allNdeNum, rgdNdeNum, elaNdeNum, ttlNum);

	return;
}
