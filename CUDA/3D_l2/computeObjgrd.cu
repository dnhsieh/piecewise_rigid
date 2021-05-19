#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "struct.h"
#include "constants.h"

void assignObjgrdStructMemory(long long &, fcndata &, double *);
void objgrd(double *, double *, double *, fcndata &);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	mxInitGPU();

	fcndata fcnObj = {0};

	mxGPUArray const *ctlGrpStk;
	mxGPUArray const *cenIniMat, *difIniMat, *grpNdeVec, *grpIfoMat, *wgtGrpVec, *tgtNdeMat;

	mxGPUArray       *grdGrpVec;

	ctlGrpStk              =  mxGPUCreateFromMxArray(prhs[ 0]);
	cenIniMat              =  mxGPUCreateFromMxArray(prhs[ 1]);
	difIniMat              =  mxGPUCreateFromMxArray(prhs[ 2]);
	grpNdeVec              =  mxGPUCreateFromMxArray(prhs[ 3]);
	grpIfoMat              =  mxGPUCreateFromMxArray(prhs[ 4]);
	wgtGrpVec              =  mxGPUCreateFromMxArray(prhs[ 5]);
	tgtNdeMat              =  mxGPUCreateFromMxArray(prhs[ 6]);
	fcnObj.prm.knlOrder    =             mxGetScalar(prhs[ 7]);
	fcnObj.prm.knlWidth    =             mxGetScalar(prhs[ 8]);
	fcnObj.prm.knlEps      =             mxGetScalar(prhs[ 9]);
	fcnObj.prm.timeStp     =             mxGetScalar(prhs[10]);
	fcnObj.prm.timeNum     =             mxGetScalar(prhs[11]);
	fcnObj.prm.tgtWgt      =             mxGetScalar(prhs[12]);

	fcnObj.prm.rgdGrpNum = mxGPUGetNumberOfElements(wgtGrpVec);

	mwSize const ndim = 1;
	mwSize const dims[1] = {(mwSize) fcnObj.prm.rgdGrpNum * RGDDOF * (fcnObj.prm.timeNum - 1)};
	grdGrpVec = mxGPUCreateGPUArray(ndim, dims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	
	// ---

	double *d_ctlGrpStk = (double *) mxGPUGetDataReadOnly(ctlGrpStk);

	fcnObj.prm.d_cenIniMat    = (double *) mxGPUGetDataReadOnly(cenIniMat);
	fcnObj.prm.d_difIniMat    = (double *) mxGPUGetDataReadOnly(difIniMat);
	fcnObj.prm.d_grpNdeVec    = (int    *) mxGPUGetDataReadOnly(grpNdeVec);
	fcnObj.prm.d_grpIfoMat    = (int    *) mxGPUGetDataReadOnly(grpIfoMat);
	fcnObj.prm.d_wgtGrpVec    = (double *) mxGPUGetDataReadOnly(wgtGrpVec);
	fcnObj.tgt.d_tgtNdeMat    = (double *) mxGPUGetDataReadOnly(tgtNdeMat);

	double *d_grdGrpVec = (double *) mxGPUGetData(grdGrpVec);

	fcnObj.prm.rgdNdeNum = mxGPUGetNumberOfElements(grpNdeVec);

	// ---

	int rgdGrpNum = fcnObj.prm.rgdGrpNum;
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;
	int timeNum   = fcnObj.prm.timeNum;

	long long gpuAloDblMemCnt =  rgdGrpNum * (timeNum - 1)
	                           + rgdNdeNum * (  rgdNdeNum * 2 + DIMNUM + DIMNUM * timeNum + DIMNUM * (timeNum - 1) * 2
	                                          + RGDDOF * (timeNum - 1) + RGDDOF * timeNum)
	                           + rgdNdeNum * (DIMNUM + RGDDOF * (timeNum - 1) + RGDDOF * 5)
	                           + SUMBLKDIM;

	double *gpuDblSpace;
	cudaError_t error = cudaMalloc((void **) &gpuDblSpace, sizeof(double) * gpuAloDblMemCnt);
	if ( error != cudaSuccess )
		mexErrMsgIdAndTxt("objgrd3D:cudaMalloc", "Fail to allocate device memory.");

	cudaMalloc((void **) &(fcnObj.d_status), sizeof(int));

	long long gpuAsgDblMemCnt;
	assignObjgrdStructMemory(gpuAsgDblMemCnt, fcnObj, gpuDblSpace);
	if ( gpuAsgDblMemCnt != gpuAloDblMemCnt )
	{
		mexErrMsgIdAndTxt("objgrd3D:memAssign", 
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
	objgrd(&h_objVal, d_grdGrpVec, d_ctlGrpStk, fcnObj);

	plhs[0] = mxCreateDoubleScalar(h_objVal);
	plhs[1] = mxGPUCreateMxArrayOnGPU(grdGrpVec);

	// ---
	//

	mxGPUDestroyGPUArray(ctlGrpStk);
	mxGPUDestroyGPUArray(cenIniMat);
	mxGPUDestroyGPUArray(difIniMat);
	mxGPUDestroyGPUArray(grpNdeVec);
	mxGPUDestroyGPUArray(grpIfoMat);
	mxGPUDestroyGPUArray(wgtGrpVec);
	mxGPUDestroyGPUArray(tgtNdeMat);
	mxGPUDestroyGPUArray(grdGrpVec);

	cudaFree(gpuDblSpace);
	cudaFree(fcnObj.d_status);
	cudaFree(fcnObj.d_workspace);

	cublasDestroy(fcnObj.blasHdl);
	cusolverDnDestroy(fcnObj.solvHdl);

	return;
}

