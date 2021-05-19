#include <cublas_v2.h>
#include <cusolverDn.h>
#include "struct.h"
#include "constants.h"

void   group2Node(double *, double *, int *, int, int, int);
double computeControlCost(double *, double *, fcndata &);
void   computeKernel(double *, double *, fcndata &);
void   computeKernel(double *, double *, double *, fcndata &);
void   addEpsIdentity(double *, double, int);
void   cholesky(double *, fcndata &);
void   computeRigidVelocity(double *, double *, double *, fcndata &);
void   computeRigidDeformation(double *, double *, fcndata &);

void   duRigidVelocity(double *, double *, double *, fcndata &);
void   dqKernel(double *, double *, double *, double *, fcndata &);
void   dqRigidVelocity(double *, double *, double *, double *, fcndata &);
void   dqRigidDeformation(double *, double *, double *, fcndata &);
void   duControl(double *, double *, double *, fcndata &);
void   node2Group(double *, double *, int *, int, int, int);

void   vectorSum(double *, double, double *, double, double *, int);
void   vectorSum(double *, double, double *, double, double *, double, double *, int);
void   vectorSum(double *, double, double *, double, double *, double, double *, double, double *, int);
void   vectorScale(double *, double, double *, int);

void objgrd(double *h_objPtr, double *d_grdGrpStk, double *d_ctlGrpStk, fcndata &fcnObj)
{
	int    rgdGrpNum = fcnObj.prm.rgdGrpNum;
	int    rgdNdeNum = fcnObj.prm.rgdNdeNum;
	int      timeNum = fcnObj.prm.timeNum;
	double   timeStp = fcnObj.prm.timeStp;

	cudaMemset(fcnObj.d_grdNdeStk, 0, sizeof(double) * rgdNdeNum * RGDDOF * (timeNum - 1));

	group2Node(fcnObj.d_ctlNdeStk, d_ctlGrpStk, fcnObj.prm.d_grpNdeVec,
	           rgdNdeNum, rgdGrpNum, timeNum);

	cudaMemset(fcnObj.d_ctlCumStk, 0, sizeof(double) * rgdNdeNum * DIMNUM);
	cudaMemcpy(fcnObj.d_ctlCumStk + rgdNdeNum * DIMNUM, fcnObj.prm.d_cenIniMat,
	           sizeof(double) * rgdNdeNum * DIMNUM, cudaMemcpyDeviceToDevice);

	computeRigidDeformation(fcnObj.d_rgdNdeStk, fcnObj.d_ctlCumStk, fcnObj);	

	*h_objPtr = computeControlCost(d_ctlGrpStk, fcnObj.prm.d_wgtGrpVec, fcnObj);

	for ( int timeIdx = 0; timeIdx < timeNum - 1; ++timeIdx )
	{
		fcnObj.d_ctlNdeMat    = fcnObj.d_ctlNdeStk +  timeIdx      * rgdNdeNum * RGDDOF;
		fcnObj.d_ctlCumNowMat = fcnObj.d_ctlCumStk +  timeIdx      * rgdNdeNum * RGDDOF;
		fcnObj.d_ctlCumNxtMat = fcnObj.d_ctlCumStk + (timeIdx + 1) * rgdNdeNum * RGDDOF;
		fcnObj.d_rgdNdeNowMat = fcnObj.d_rgdNdeStk +  timeIdx      * rgdNdeNum * DIMNUM;
		fcnObj.d_rgdNdeNxtMat = fcnObj.d_rgdNdeStk + (timeIdx + 1) * rgdNdeNum * DIMNUM;
		fcnObj.d_rgdVlcMat    = fcnObj.d_rgdVlcStk +  timeIdx      * rgdNdeNum * DIMNUM;
		fcnObj.d_rgdAlpMat    = fcnObj.d_rgdAlpStk +  timeIdx      * rgdNdeNum * DIMNUM;

		computeRigidVelocity(fcnObj.d_rgdVlcMat, fcnObj.d_ctlNdeMat, fcnObj.d_ctlCumNowMat, fcnObj);

		computeKernel(fcnObj.d_rgdKnlMat, fcnObj.d_rgdNdeNowMat, fcnObj); 
		addEpsIdentity(fcnObj.d_rgdKnlMat, fcnObj.prm.knlEps, rgdNdeNum);

		cudaMemcpy(fcnObj.d_rgdKnLMat, fcnObj.d_rgdKnlMat,
		           sizeof(double) * rgdNdeNum * rgdNdeNum, cudaMemcpyDeviceToDevice);
		cholesky(fcnObj.d_rgdKnLMat, fcnObj);

		cudaMemcpy(fcnObj.d_rgdAlpMat, fcnObj.d_rgdVlcMat,
		           sizeof(double) * rgdNdeNum * DIMNUM, cudaMemcpyDeviceToDevice);
		cusolverDnDpotrs(fcnObj.solvHdl, CUBLAS_FILL_MODE_LOWER, rgdNdeNum, DIMNUM,
		                 fcnObj.d_rgdKnLMat, rgdNdeNum,
		                 fcnObj.d_rgdAlpMat, rgdNdeNum, fcnObj.d_status);

		double h_ldmVal;
		cublasDdot(fcnObj.blasHdl, rgdNdeNum * DIMNUM,
		           fcnObj.d_rgdAlpMat, 1, fcnObj.d_rgdVlcMat, 1, &h_ldmVal);
		*h_objPtr += 0.5 * h_ldmVal;

		vectorSum(fcnObj.d_ctlCumNxtMat,
		          1.0, fcnObj.d_ctlCumNowMat, timeStp, fcnObj.d_ctlNdeMat, rgdNdeNum * RGDDOF);
	
		computeRigidDeformation(fcnObj.d_rgdNdeNxtMat, fcnObj.d_ctlCumNxtMat, fcnObj);
	}

	double *d_rgdNdeEndMat = fcnObj.d_rgdNdeStk + (timeNum - 1) * rgdNdeNum * DIMNUM;

	vectorSum(fcnObj.d_difMat, 1.0, d_rgdNdeEndMat, -1.0, fcnObj.tgt.d_tgtNdeMat, rgdNdeNum * DIMNUM);

	double h_l2SquVal;
	cublasDdot(fcnObj.blasHdl, rgdNdeNum * DIMNUM, fcnObj.d_difMat, 1, fcnObj.d_difMat, 1, &h_l2SquVal);

	*h_objPtr = timeStp * (*h_objPtr) + fcnObj.prm.tgtWgt * 0.5 * h_l2SquVal;

	double *d_ctlCumEndMat = fcnObj.d_ctlCumStk + (timeNum - 1) * rgdNdeNum * RGDDOF;
	dqRigidDeformation(fcnObj.d_pMat, d_ctlCumEndMat, fcnObj.d_difMat, fcnObj);
	vectorScale(fcnObj.d_pMat, -fcnObj.prm.tgtWgt, fcnObj.d_pMat, rgdNdeNum * RGDDOF);

	for ( int timeIdx = timeNum - 2; timeIdx >= 0; --timeIdx )
	{
		fcnObj.d_grdNdeMat    = fcnObj.d_grdNdeStk + timeIdx * rgdNdeNum * RGDDOF;

		fcnObj.d_ctlNdeMat    = fcnObj.d_ctlNdeStk + timeIdx * rgdNdeNum * RGDDOF;
		fcnObj.d_ctlCumNowMat = fcnObj.d_ctlCumStk + timeIdx * rgdNdeNum * RGDDOF;
		fcnObj.d_rgdNdeNowMat = fcnObj.d_rgdNdeStk + timeIdx * rgdNdeNum * DIMNUM;
		fcnObj.d_rgdAlpMat    = fcnObj.d_rgdAlpStk + timeIdx * rgdNdeNum * DIMNUM;

		duRigidVelocity(fcnObj.d_duRVMat, fcnObj.d_ctlCumNowMat, fcnObj.d_rgdAlpMat, fcnObj);
		vectorSum(fcnObj.d_grdNdeMat, 1.0, fcnObj.d_duRVMat, -1.0, fcnObj.d_pMat, rgdNdeNum * RGDDOF);

		dqKernel(fcnObj.d_dqKMat, fcnObj.d_rgdNdeNowMat, fcnObj.d_rgdAlpMat, fcnObj.d_rgdAlpMat, fcnObj);
		vectorScale(fcnObj.d_dqKMat, -0.5, fcnObj.d_dqKMat, rgdNdeNum * DIMNUM);

		dqRigidDeformation(fcnObj.d_dqRDMat, fcnObj.d_ctlCumNowMat, fcnObj.d_dqKMat, fcnObj);
		dqRigidVelocity(fcnObj.d_dqRVMat, fcnObj.d_ctlNdeMat, fcnObj.d_ctlCumNowMat,
		                fcnObj.d_rgdAlpMat, fcnObj);

		vectorSum(fcnObj.d_pDotMat, 1.0, fcnObj.d_dqRDMat, 1.0,     fcnObj.d_dqRVMat, rgdNdeNum * RGDDOF);
		vectorSum(fcnObj.d_pMat,    1.0, fcnObj.d_pMat,   -timeStp, fcnObj.d_pDotMat, rgdNdeNum * RGDDOF);
	}

	node2Group(d_grdGrpStk, fcnObj.d_grdNdeStk, fcnObj.prm.d_grpIfoMat,
	           rgdNdeNum, rgdGrpNum, timeNum);
	duControl(d_grdGrpStk, d_ctlGrpStk, fcnObj.prm.d_wgtGrpVec, fcnObj);
	vectorScale(d_grdGrpStk, timeStp, d_grdGrpStk, rgdGrpNum * RGDDOF * (timeNum - 1));

	return;
}
