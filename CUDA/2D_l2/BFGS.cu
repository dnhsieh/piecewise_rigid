// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 06/10/2020

#include <cstdio>
#include <cmath>
#include <cublas_v2.h>
#include "mex.h"
#include "struct.h"
#include "constants.h"

void objgrd(double *, double *, double *, fcndata &);

int  lineSearch(double *, double *, double *, double *, double, double, double, 
                double *, double *, double *, double &, int &, fcndata &);
void updateInverseHessian(double *, double *, double *, double *, int, cublasHandle_t);

__global__ void uniformDiagKernel(double *d_AMat, double fillVal, int varNum)
{
	int varIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( varIdx < varNum )
		d_AMat[varIdx * varNum + varIdx] = fillVal;

	return;
}

void uniformDiag(double *d_AMat, double fillVal, int varNum)
{
	cudaMemset(d_AMat, 0, sizeof(double) * varNum * varNum);

	int blkNum = (varNum - 1) / BLKDIM + 1;
	uniformDiagKernel <<<blkNum, BLKDIM>>> (d_AMat, fillVal, varNum);

	return;
}

__global__ void vectorSubtractKernel(double *d_v12Vec, double *d_v1Vec, double *d_v2Vec, int varNum)
{
	int varIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( varIdx < varNum )
		d_v12Vec[varIdx] = d_v1Vec[varIdx] - d_v2Vec[varIdx];

	return;
}

void vectorSubtract(double *d_v12Vec, double *d_v1Vec, double *d_v2Vec, int varNum)
{
	int blkNum = (varNum - 1) / BLKDIM + 1;
	vectorSubtractKernel <<<blkNum, BLKDIM>>> (d_v12Vec, d_v1Vec, d_v2Vec, varNum);

	return;
}

int BFGS(double *h_fcnNow, double *d_grdNow, double *d_posNow, optdata &optObj, fcndata &fcnObj)
{
	int     varNum = optObj.varNum;
	int     itrMax = optObj.itrMax;
	double  tolVal = optObj.tolVal;
	double  wolfe1 = optObj.wolfe1;
	double  wolfe2 = optObj.wolfe2;
	bool    vbsFlg = optObj.vbsFlg;

	double *d_apHMat = optObj.d_apHMat;
	double *d_dirVec = optObj.d_dirVec;
	double *d_posNxt = optObj.d_posNxt;
	double *d_grdNxt = optObj.d_grdNxt;
	double *d_dspVec = optObj.d_dspVec;
	double *d_dgdVec = optObj.d_dgdVec;
	double *d_tmpVec = optObj.d_tmpVec;
	double  h_fcnNxt;

	objgrd(h_fcnNow, d_grdNow, d_posNow, fcnObj);

	double h_grdSqu;
	cublasDdot(fcnObj.blasHdl, varNum, d_grdNow, 1, d_grdNow, 1, &h_grdSqu);
	double h_grdLen = sqrt(h_grdSqu);

	if ( vbsFlg )
	{
		mexPrintf("%5s   %13s  %13s  %13s  %9s\n", "iter", "f", "|grad f|", "step length", "fcn eval");
		char sepStr[65] = {0};
		memset(sepStr, '-', 62);
		mexPrintf("%s\n", sepStr);
		mexPrintf("%5d:  %13.6e  %13.6e\n", 0, *h_fcnNow, h_grdLen);
	}

	uniformDiag(d_apHMat, 1.0, varNum);

	for ( int itrIdx = 1; itrIdx <= itrMax; ++itrIdx )
	{
		if ( h_grdLen < tolVal )
			break;

		if ( itrIdx == 2 )
		{
			double h_dspDgd, h_dgdDgd;
			cublasDdot(fcnObj.blasHdl, varNum, d_dspVec, 1, d_dgdVec, 1, &h_dspDgd);
			cublasDdot(fcnObj.blasHdl, varNum, d_dgdVec, 1, d_dgdVec, 1, &h_dgdDgd);

			uniformDiag(d_apHMat, h_dspDgd / h_dgdDgd, varNum);
		}

		double alpVal = -1.0, btaVal = 0.0;
		cublasDgemv(fcnObj.blasHdl, CUBLAS_OP_N, varNum, varNum, &alpVal, d_apHMat, varNum,
		            d_grdNow, 1, &btaVal, d_dirVec, 1);

		double stpLen;
		int    objCnt;
		int    lineErr = lineSearch(d_posNow, d_grdNow, d_dirVec, h_fcnNow, wolfe1, wolfe2, tolVal, 
		                            d_posNxt, d_grdNxt, &h_fcnNxt, stpLen, objCnt, fcnObj);
		if ( lineErr != 0 ) return 1;

		vectorSubtract(d_dspVec, d_posNxt, d_posNow, varNum);
		vectorSubtract(d_dgdVec, d_grdNxt, d_grdNow, varNum);

		if ( itrIdx >= 2 )
			updateInverseHessian(d_apHMat, d_dspVec, d_dgdVec, d_tmpVec, varNum, fcnObj.blasHdl);

		cudaMemcpy(d_posNow, d_posNxt, sizeof(double) * varNum, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_grdNow, d_grdNxt, sizeof(double) * varNum, cudaMemcpyDeviceToDevice);
		*h_fcnNow = h_fcnNxt;
		cublasDdot(fcnObj.blasHdl, varNum, d_grdNow, 1, d_grdNow, 1, &h_grdSqu);
		h_grdLen = sqrt(h_grdSqu);

		if ( vbsFlg )
			mexPrintf("%5d:  %13.6e  %13.6e  %13.6e  %9d\n", itrIdx, *h_fcnNow, h_grdLen, stpLen, objCnt);
	}

	return 0;
}
