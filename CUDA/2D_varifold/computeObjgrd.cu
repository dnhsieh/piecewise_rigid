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
	mxGPUArray const *cenIniMat, *difIniMat, *grpNdeVec, *grpIfoMat, *wgtGrpVec;
	mxGPUArray const *vfdElmVtxMat, *vfdElmIfoMat;
	mxGPUArray const *tgtCenPosMat, *tgtUniDirMat, *tgtElmVolVec;

	mxGPUArray       *grdGrpVec;

	ctlGrpStk              =  mxGPUCreateFromMxArray(prhs[ 0]);
	cenIniMat              =  mxGPUCreateFromMxArray(prhs[ 1]);
	difIniMat              =  mxGPUCreateFromMxArray(prhs[ 2]);
	grpNdeVec              =  mxGPUCreateFromMxArray(prhs[ 3]);
	grpIfoMat              =  mxGPUCreateFromMxArray(prhs[ 4]);
	wgtGrpVec              =  mxGPUCreateFromMxArray(prhs[ 5]);
	vfdElmVtxMat           =  mxGPUCreateFromMxArray(prhs[ 6]);
	vfdElmIfoMat           =  mxGPUCreateFromMxArray(prhs[ 7]);
	tgtCenPosMat           =  mxGPUCreateFromMxArray(prhs[ 8]);
	tgtUniDirMat           =  mxGPUCreateFromMxArray(prhs[ 9]);
	tgtElmVolVec           =  mxGPUCreateFromMxArray(prhs[10]);
	fcnObj.vfd.cenKnlType  =             mxGetScalar(prhs[11]);
	fcnObj.vfd.cenKnlWidth =             mxGetScalar(prhs[12]);
	fcnObj.vfd.dirKnlType  =             mxGetScalar(prhs[13]);
	fcnObj.vfd.dirKnlWidth =             mxGetScalar(prhs[14]);
	fcnObj.prm.knlOrder    =             mxGetScalar(prhs[15]);
	fcnObj.prm.knlWidth    =             mxGetScalar(prhs[16]);
	fcnObj.prm.knlEps      =             mxGetScalar(prhs[17]);
	fcnObj.prm.timeStp     =             mxGetScalar(prhs[18]);
	fcnObj.prm.timeNum     =             mxGetScalar(prhs[19]);
	fcnObj.prm.tgtWgt      =             mxGetScalar(prhs[20]);

	fcnObj.prm.rgdGrpNum = mxGPUGetNumberOfElements(wgtGrpVec);

	mwSize const ndim = 1;
	mwSize const grdDims[1] = {(mwSize) fcnObj.prm.rgdGrpNum * RGDDOF * (fcnObj.prm.timeNum - 1)};
	grdGrpVec = mxGPUCreateGPUArray(ndim, grdDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	
	// ---

	double *d_ctlGrpStk = (double *) mxGPUGetDataReadOnly(ctlGrpStk);

	fcnObj.prm.d_cenIniMat    = (double *) mxGPUGetDataReadOnly(cenIniMat);
	fcnObj.prm.d_difIniMat    = (double *) mxGPUGetDataReadOnly(difIniMat);
	fcnObj.prm.d_grpNdeVec    = (int    *) mxGPUGetDataReadOnly(grpNdeVec);
	fcnObj.prm.d_grpIfoMat    = (int    *) mxGPUGetDataReadOnly(grpIfoMat);
	fcnObj.prm.d_wgtGrpVec    = (double *) mxGPUGetDataReadOnly(wgtGrpVec);
	fcnObj.elm.d_vfdElmVtxMat = (int    *) mxGPUGetDataReadOnly(vfdElmVtxMat);
	fcnObj.elm.d_vfdElmIfoMat = (int    *) mxGPUGetDataReadOnly(vfdElmIfoMat);
	fcnObj.tgt.d_cenPosMat    = (double *) mxGPUGetDataReadOnly(tgtCenPosMat);
	fcnObj.tgt.d_uniDirMat    = (double *) mxGPUGetDataReadOnly(tgtUniDirMat);
	fcnObj.tgt.d_elmVolVec    = (double *) mxGPUGetDataReadOnly(tgtElmVolVec);

	double *d_grdGrpVec = (double *) mxGPUGetData(grdGrpVec);

	mwSize const *vfdElmDims = mxGPUGetDimensions(vfdElmVtxMat);
	mwSize const *tgtElmDims = mxGPUGetDimensions(tgtCenPosMat);

	fcnObj.prm.rgdNdeNum = mxGPUGetNumberOfElements(grpNdeVec);
	fcnObj.prm.vfdNdeNum = fcnObj.prm.rgdNdeNum;
	fcnObj.prm.vfdElmNum = vfdElmDims[0];
	fcnObj.tgt.tgtElmNum = tgtElmDims[0];

	// ---

	int rgdGrpNum = fcnObj.prm.rgdGrpNum;
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;
	int vfdElmNum = fcnObj.prm.vfdElmNum;
	int timeNum   = fcnObj.prm.timeNum;

	long long gpuAloDblMemCnt =  rgdGrpNum * (timeNum - 1)
	                           + rgdNdeNum * (  rgdNdeNum * 2 + DIMNUM * timeNum + DIMNUM * (timeNum - 1) * 2
	                                          + RGDDOF * (timeNum - 1) + RGDDOF * timeNum)
	                           + vfdElmNum * (DIMNUM * 2 + 2) + fcnObj.tgt.tgtElmNum 
	                           + rgdNdeNum * (DIMNUM * 2 + RGDDOF * (timeNum - 1) + RGDDOF * 5)
	                           + vfdElmNum * DIMNUM * 2
	                           + SUMBLKDIM;

	double *gpuDblSpace;
	cudaError_t error = cudaMalloc((void **) &gpuDblSpace, sizeof(double) * gpuAloDblMemCnt);
	if ( error != cudaSuccess )
		mexErrMsgIdAndTxt("objgrd2D:cudaMalloc", "Fail to allocate device memory.");

	cudaMalloc((void **) &(fcnObj.d_status), sizeof(int));

	long long gpuAsgDblMemCnt;
	assignObjgrdStructMemory(gpuAsgDblMemCnt, fcnObj, gpuDblSpace);
	if ( gpuAsgDblMemCnt != gpuAloDblMemCnt )
	{
		mexErrMsgIdAndTxt("objgrd2D:memAssign", 
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
	mxGPUDestroyGPUArray(vfdElmVtxMat);
	mxGPUDestroyGPUArray(vfdElmIfoMat);
	mxGPUDestroyGPUArray(tgtCenPosMat);
	mxGPUDestroyGPUArray(tgtUniDirMat);
	mxGPUDestroyGPUArray(tgtElmVolVec);
	mxGPUDestroyGPUArray(grdGrpVec);

	mxFree((void *) vfdElmDims);
	mxFree((void *) tgtElmDims);

	cudaFree(gpuDblSpace);
	cudaFree(fcnObj.d_status);
	cudaFree(fcnObj.d_workspace);

	cublasDestroy(fcnObj.blasHdl);
	cusolverDnDestroy(fcnObj.solvHdl);

	return;
}

