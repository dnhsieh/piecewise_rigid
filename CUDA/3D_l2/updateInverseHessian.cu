// Equation (6.17) in Nocedal
//
// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 05/20/2020

#include <cublas_v2.h>
#include "constants.h"

__global__ void updateKernel(double *d_HMat, double *d_sVec, double *d_HyVec, 
                             double rhoVal, double yHyVal, int varNum)
{
	__shared__ double  s_sRowVec[BLKROW];
	__shared__ double  s_sColVec[BLKROW];
	__shared__ double s_HyRowVec[BLKROW];
	__shared__ double s_HyColVec[BLKROW];

	int rowBgn = blockIdx.x * blockDim.x;
	int colBgn = blockIdx.y * blockDim.y;
	if ( threadIdx.y == 0 )
	{
		int rowIdx = rowBgn + threadIdx.x;
		s_sRowVec[threadIdx.x] = (rowIdx < varNum ? d_sVec[rowIdx] : 0.0);
	}
	else if ( threadIdx.y == 2 )
	{
		int rowIdx = rowBgn + threadIdx.x;
		s_HyRowVec[threadIdx.x] = (rowIdx < varNum ? d_HyVec[rowIdx] : 0.0);
	}
	else if ( threadIdx.y == 4 )
	{
		int colIdx = colBgn + threadIdx.x;
		s_sColVec[threadIdx.x] = (colIdx < varNum ? d_sVec[colIdx] : 0.0);
	}
	else if ( threadIdx.y == 6 )
	{
		int colIdx = colBgn + threadIdx.x;
		s_HyColVec[threadIdx.x] = (colIdx < varNum ? d_HyVec[colIdx] : 0.0);
	}
	__syncthreads();

	int rowIdx = rowBgn + threadIdx.x;
	int colIdx = colBgn + threadIdx.y;
	if ( rowIdx < varNum && colIdx < varNum )
	{
		d_HMat[colIdx * varNum + rowIdx] +=
		   rhoVal * (  -s_sColVec[threadIdx.y] * s_HyRowVec[threadIdx.x]
		             - s_HyColVec[threadIdx.y] *  s_sRowVec[threadIdx.x]
		             + (rhoVal * yHyVal + 1.0) *  s_sColVec[threadIdx.y] * s_sRowVec[threadIdx.x]);
	}

	return;
}

void updateInverseHessian(double *d_HMat, double *d_sVec, double *d_yVec, double *d_HyVec,
                          int varNum, cublasHandle_t blasHdl)
{
	// s   = x_next - n_now = dspVec
	// y   = (grad f)_next - (grad f)_now = dgdVec
	// rho = 1 / (s^T y)

	double h_rhoInv;
	cublasDdot(blasHdl, varNum, d_sVec, 1, d_yVec, 1, &h_rhoInv);
	double h_rhoVal = 1.0 / h_rhoInv;

	double alpVal = 1.0, btaVal = 0.0;
	cublasDgemv(blasHdl, CUBLAS_OP_N, varNum, varNum, &alpVal, d_HMat, varNum,
	            d_yVec, 1, &btaVal, d_HyVec, 1);

	double h_yHyVal;
	cublasDdot(blasHdl, varNum, d_yVec, 1, d_HyVec, 1, &h_yHyVal);

	int  gridRow = (varNum - 1) / BLKROW + 1;
	dim3 blkNum(gridRow, gridRow);
	dim3 blkDim( BLKROW,  BLKROW);
	updateKernel <<<blkNum, blkDim>>> (d_HMat, d_sVec, d_HyVec, h_rhoVal, h_yHyVal, varNum);

	return;
}
