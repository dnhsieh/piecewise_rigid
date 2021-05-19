#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "struct.h"
#include "constants.h"

void assignObjfcnStructMemory(long long &, fcndata &, double *);
void objfcn(double *, double *, fcndata &);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	mxInitGPU();

	fcndata fcnObj = {0};

	mxGPUArray const *ctlGrpStk;
	mxGPUArray const *cenIniMat, *difIniMat, *grpNdeVec, *wgtGrpVec, *tgtNdeMat;

	ctlGrpStk           =  mxGPUCreateFromMxArray(prhs[ 0]);
	cenIniMat           =  mxGPUCreateFromMxArray(prhs[ 1]);
	difIniMat           =  mxGPUCreateFromMxArray(prhs[ 2]);
	grpNdeVec           =  mxGPUCreateFromMxArray(prhs[ 3]);
	wgtGrpVec           =  mxGPUCreateFromMxArray(prhs[ 4]);
	tgtNdeMat           =  mxGPUCreateFromMxArray(prhs[ 5]);
	fcnObj.prm.knlOrder =             mxGetScalar(prhs[ 6]);
	fcnObj.prm.knlWidth =             mxGetScalar(prhs[ 7]);
	fcnObj.prm.knlEps   =             mxGetScalar(prhs[ 8]);
	fcnObj.prm.timeStp  =             mxGetScalar(prhs[ 9]);
	fcnObj.prm.timeNum  =             mxGetScalar(prhs[10]);
	fcnObj.prm.tgtWgt   =             mxGetScalar(prhs[11]);

	// ---

	double *d_ctlGrpStk = (double *) mxGPUGetDataReadOnly(ctlGrpStk);

	fcnObj.prm.d_cenIniMat    = (double *) mxGPUGetDataReadOnly(cenIniMat);
	fcnObj.prm.d_difIniMat    = (double *) mxGPUGetDataReadOnly(difIniMat);
	fcnObj.prm.d_grpNdeVec    = (int    *) mxGPUGetDataReadOnly(grpNdeVec);
	fcnObj.prm.d_wgtGrpVec    = (double *) mxGPUGetDataReadOnly(wgtGrpVec);
	fcnObj.tgt.d_tgtNdeMat    = (double *) mxGPUGetDataReadOnly(tgtNdeMat);

	fcnObj.prm.rgdGrpNum = mxGPUGetNumberOfElements(wgtGrpVec);
	fcnObj.prm.rgdNdeNum = mxGPUGetNumberOfElements(grpNdeVec);

	// ---

	int rgdGrpNum = fcnObj.prm.rgdGrpNum;
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;
	int timeNum   = fcnObj.prm.timeNum;

	long long gpuAloDblMemCnt =  rgdGrpNum * (timeNum - 1)
	                           + rgdNdeNum * (  rgdNdeNum * 2 + DIMNUM + DIMNUM * timeNum + DIMNUM * (timeNum - 1) * 2
	                                          + RGDDOF * (timeNum - 1) + RGDDOF * timeNum)
	                           + SUMBLKDIM;

	double *gpuDblSpace;
	cudaError_t error = cudaMalloc((void **) &gpuDblSpace, sizeof(double) * gpuAloDblMemCnt);
	if ( error != cudaSuccess )
		mexErrMsgIdAndTxt("objfcn2D:cudaMalloc", "Fail to allocate device memory.");

	cudaMalloc((void **) &(fcnObj.d_status), sizeof(int));

	long long gpuAsgDblMemCnt;
	assignObjfcnStructMemory(gpuAsgDblMemCnt, fcnObj, gpuDblSpace);
	if ( gpuAsgDblMemCnt != gpuAloDblMemCnt )
	{
		mexErrMsgIdAndTxt("objfcn2D:memAssign", 
		                  "Assigned device double memory (%lld) mismatches the allocated memory (%lld).", 
		                  gpuAsgDblMemCnt, gpuAloDblMemCnt);
	}

	// ---

	cublasCreate(&(fcnObj.blasHdl));

	cusolverDnCreate(&(fcnObj.solvHdl));
	cusolverDnDpotrf_bufferSize(fcnObj.solvHdl, CUBLAS_FILL_MODE_LOWER,
	                            fcnObj.prm.rgdNdeNum, fcnObj.d_rgdKnlMat,
	                            fcnObj.prm.rgdNdeNum, &(fcnObj.h_Lwork));

	cudaMalloc((void **) &(fcnObj.d_workspace), sizeof(double) * fcnObj.h_Lwork);

	// ---

	double h_objVal;
	objfcn(&h_objVal, d_ctlGrpStk, fcnObj);

	plhs[0] = mxCreateDoubleScalar(h_objVal);

	// ---
	//

	mxGPUDestroyGPUArray(ctlGrpStk);
	mxGPUDestroyGPUArray(cenIniMat);
	mxGPUDestroyGPUArray(difIniMat);
	mxGPUDestroyGPUArray(grpNdeVec);
	mxGPUDestroyGPUArray(wgtGrpVec);
	mxGPUDestroyGPUArray(tgtNdeMat);

	cudaFree(gpuDblSpace);
	cudaFree(fcnObj.d_status);
	cudaFree(fcnObj.d_workspace);

	cublasDestroy(fcnObj.blasHdl);
	cusolverDnDestroy(fcnObj.solvHdl);

	return;
}

