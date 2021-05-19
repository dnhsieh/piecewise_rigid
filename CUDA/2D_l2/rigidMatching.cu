#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "struct.h"
#include "constants.h"

void assignOptimizationStructMemory(long long &, optdata &, double *);
void assignObjgrdStructMemory(long long &, fcndata &, double *);
int  BFGS(double *, double *, double *, optdata &, fcndata &);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	mxInitGPU();

	optdata optObj = {0};
	fcndata fcnObj = {0};

	mxGPUArray const *optIniStk;
	mxGPUArray const *cenIniMat, *difIniMat, *grpNdeVec, *grpIfoMat, *wgtGrpVec, *tgtNdeMat;

	mxGPUArray       *posNowStk, *grdNowStk;

	optIniStk              =  mxGPUCreateFromMxArray(prhs[ 0]);
	optObj.itrMax          =             mxGetScalar(prhs[ 1]);
	optObj.tolVal          =             mxGetScalar(prhs[ 2]);
	optObj.wolfe1          =             mxGetScalar(prhs[ 3]);
	optObj.wolfe2          =             mxGetScalar(prhs[ 4]);
	optObj.vbsFlg          =             mxGetScalar(prhs[ 5]);
	cenIniMat              =  mxGPUCreateFromMxArray(prhs[ 6]);
	difIniMat              =  mxGPUCreateFromMxArray(prhs[ 7]);
	grpNdeVec              =  mxGPUCreateFromMxArray(prhs[ 8]);
	grpIfoMat              =  mxGPUCreateFromMxArray(prhs[ 9]);
	wgtGrpVec              =  mxGPUCreateFromMxArray(prhs[10]);
	tgtNdeMat              =  mxGPUCreateFromMxArray(prhs[11]);
	fcnObj.prm.knlOrder    =             mxGetScalar(prhs[12]);
	fcnObj.prm.knlWidth    =             mxGetScalar(prhs[13]);
	fcnObj.prm.knlEps      =             mxGetScalar(prhs[14]);
	fcnObj.prm.timeStp     =             mxGetScalar(prhs[15]);
	fcnObj.prm.timeNum     =             mxGetScalar(prhs[16]);
	fcnObj.prm.tgtWgt      =             mxGetScalar(prhs[17]);

	fcnObj.prm.rgdGrpNum = mxGPUGetNumberOfElements(wgtGrpVec);

	mwSize const ndim = 3;
	mwSize const dims[3] = {(mwSize) fcnObj.prm.rgdGrpNum, (mwSize) RGDDOF, (mwSize) (fcnObj.prm.timeNum - 1)};
	posNowStk = mxGPUCreateGPUArray(ndim, dims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	grdNowStk = mxGPUCreateGPUArray(ndim, dims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	
	// ---

	double *d_optIniStk = (double *) mxGPUGetDataReadOnly(optIniStk);

	fcnObj.prm.d_cenIniMat    = (double *) mxGPUGetDataReadOnly(cenIniMat);
	fcnObj.prm.d_difIniMat    = (double *) mxGPUGetDataReadOnly(difIniMat);
	fcnObj.prm.d_grpNdeVec    = (int    *) mxGPUGetDataReadOnly(grpNdeVec);
	fcnObj.prm.d_grpIfoMat    = (int    *) mxGPUGetDataReadOnly(grpIfoMat);
	fcnObj.prm.d_wgtGrpVec    = (double *) mxGPUGetDataReadOnly(wgtGrpVec);
	fcnObj.tgt.d_tgtNdeMat    = (double *) mxGPUGetDataReadOnly(tgtNdeMat);

	double  h_objVal;
	double *d_posNowStk = (double *) mxGPUGetData(posNowStk);
	double *d_grdNowStk = (double *) mxGPUGetData(grdNowStk);

	fcnObj.prm.rgdNdeNum = mxGPUGetNumberOfElements(grpNdeVec);

	int optVarNum = mxGPUGetNumberOfElements(optIniStk);
	optObj.varNum     = optVarNum;
	fcnObj.prm.varNum = optVarNum;

	// ---

	int rgdGrpNum = fcnObj.prm.rgdGrpNum;
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;
	int timeNum   = fcnObj.prm.timeNum;

	long long optAloDblMemCnt =  optVarNum * (optVarNum + 6);
	long long gpuAloDblMemCnt =  rgdGrpNum * (timeNum - 1)
	                           + rgdNdeNum * (  rgdNdeNum * 2 + DIMNUM + DIMNUM * timeNum + DIMNUM * (timeNum - 1) * 2
	                                          + RGDDOF * (timeNum - 1) + RGDDOF * timeNum)
	                           + rgdNdeNum * (DIMNUM + RGDDOF * (timeNum - 1) + RGDDOF * 5)
	                           + SUMBLKDIM;

	double *optDblSpace;
	cudaError_t error = cudaMalloc((void **) &optDblSpace, sizeof(double) * optAloDblMemCnt);
	if ( error != cudaSuccess )
		mexErrMsgIdAndTxt("rigidMatching2D:cudaMalloc", "Fail to allocate device memory.");

	double *gpuDblSpace;
	error = cudaMalloc((void **) &gpuDblSpace, sizeof(double) * gpuAloDblMemCnt);
	if ( error != cudaSuccess )
		mexErrMsgIdAndTxt("rigidMatching2D:cudaMalloc", "Fail to allocate device memory.");

	cudaMalloc((void **) &(fcnObj.d_status), sizeof(int));

	long long optAsgDblMemCnt;
	assignOptimizationStructMemory(optAsgDblMemCnt, optObj, optDblSpace);
	if ( optAsgDblMemCnt != optAloDblMemCnt )
	{
		mexErrMsgIdAndTxt("rigidMatching2D:memAssign", 
		                  "Assigned device double memory (%lld) mismatches the allocated memory (%lld).", 
		                  optAsgDblMemCnt, optAloDblMemCnt);
	}

	long long gpuAsgDblMemCnt;
	assignObjgrdStructMemory(gpuAsgDblMemCnt, fcnObj, gpuDblSpace);
	if ( gpuAsgDblMemCnt != gpuAloDblMemCnt )
	{
		mexErrMsgIdAndTxt("rigidMatching2D:memAssign", 
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

	cudaMemcpy(d_posNowStk, d_optIniStk, sizeof(double) * optVarNum, cudaMemcpyDeviceToDevice);
	BFGS(&h_objVal, d_grdNowStk, d_posNowStk, optObj, fcnObj);

	plhs[0] =    mxCreateDoubleScalar(h_objVal );
	plhs[1] = mxGPUCreateMxArrayOnGPU(posNowStk);
	plhs[2] = mxGPUCreateMxArrayOnGPU(grdNowStk);

	// ---
	//

	mxGPUDestroyGPUArray(optIniStk);
	mxGPUDestroyGPUArray(cenIniMat);
	mxGPUDestroyGPUArray(difIniMat);
	mxGPUDestroyGPUArray(grpNdeVec);
	mxGPUDestroyGPUArray(grpIfoMat);
	mxGPUDestroyGPUArray(wgtGrpVec);
	mxGPUDestroyGPUArray(tgtNdeMat);
	mxGPUDestroyGPUArray(posNowStk);
	mxGPUDestroyGPUArray(grdNowStk);

	cudaFree(optDblSpace);
	cudaFree(gpuDblSpace);
	cudaFree(fcnObj.d_status);
	cudaFree(fcnObj.d_workspace);

	cublasDestroy(fcnObj.blasHdl);
	cusolverDnDestroy(fcnObj.solvHdl);

	return;
}

