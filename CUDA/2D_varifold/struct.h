#ifndef STRUCT_H
#define STRUCT_H

#include <cusolverDn.h>
#include <cublas_v2.h>

struct parameters
{
	int     varNum;

	int     rgdGrpNum;
	int     rgdNdeNum;
	int     vfdNdeNum;
	int     vfdElmNum;

	double *d_cenIniMat;
	double *d_difIniMat;
	int    *d_grpNdeVec;
	int    *d_grpIfoMat;
	double *d_wgtGrpVec;
	int     knlOrder;
	double  knlWidth;
	double  knlEps;
	double  timeStp;
	int     timeNum;
	double  tgtWgt;
};

struct element
{
	int *d_vfdElmVtxMat;
	int *d_vfdElmIfoMat;
};

struct target
{
	int tgtElmNum;

	double *d_cenPosMat;
	double *d_uniDirMat;
	double *d_elmVolVec;
};

struct varifold
{
	char   cenKnlType;
	double cenKnlWidth;
	char   dirKnlType;
	double dirKnlWidth;
};

struct fcndata
{
	struct parameters prm;
	struct element    elm;
	struct target     tgt;
	struct varifold   vfd;

	double *d_ctlNdeStk;       // rgdNdeNum * RGDDOF * (timeNum - 1)
	double *d_ctlNdeMat;       // no memory allocation, pointing to the data
	double *d_ctlCumStk;       // rgdNdeNum * RGDDOF * timeNum
	double *d_ctlCumNowMat;    // no memory allocation, pointing to the data
	double *d_ctlCumNxtMat;    // no memory allocation, pointing to the data
	double *d_ctlVec;          // rgdGrpNum * (timeNum - 1)
	double *d_rgdNdeStk;       // rgdNdeNum * DIMNUM * timeNum
	double *d_rgdNdeNowMat;    // no memory allocation, pointing to the data
	double *d_rgdNdeNxtMat;    // no memory allocation, pointing to the data
	double *d_rgdVlcStk;       // rgdNdeNum * DIMNUM * (timeNum - 1)
	double *d_rgdVlcMat;       // no memory allocation, pointing to the data
	double *d_rgdAlpStk;       // rgdNdeNum * DIMNUM * (timeNum - 1)
	double *d_rgdAlpMat;       // no memory allocation, pointing to the data
	double *d_rgdKnlMat;       // rgdNdeNum * rgdNdeNum
	double *d_rgdKnLMat;       // rgdNdeNum * rgdNdeNum 

	double *d_dfmCenPosMat;    // vfdElmNum * DIMNUM
	double *d_dfmUniDirMat;    // vfdElmNum * DIMNUM
	double *d_dfmElmVolVec;    // vfdElmNum
	double *d_vfdVec;          // vfdElmNum + tgtElmNum
	double *d_dqVfdMat;        // rgdNdeNum * DIMNUM
	double *d_dcVfdMat;        // vfdElmNum * DIMNUM
	double *d_ddVfdMat;        // vfdElmNum * DIMNUM

	double *d_grdNdeStk;       // rgdNdeNum * RGDDOF * (timeNum - 1)
	double *d_grdNdeMat;       // no memory allocation, pointing to the data
	double *d_pMat;            // rgdNdeNum * RGDDOF 
	double *d_pDotMat;         // rgdNdeNum * RGDDOF
	double *d_duRVMat;         // rgdNdeNum * RGDDOF
	double *d_dqKMat;          // rgdNdeNum * DIMNUM
	double *d_dqRVMat;         // rgdNdeNum * RGDDOF
	double *d_dqRDMat;         // rgdNdeNum * RGDDOF

	double *d_sumBufVec;       // SUMBLKDIM

	cublasHandle_t     blasHdl;
	cusolverDnHandle_t solvHdl;

	int     h_Lwork;
	double *d_workspace;       // sizeof(double) * h_Lwork
	int    *d_status;          // sizeof(int) * 1
};

struct optdata
{
	int    varNum;

	int    itrMax;
	double tolVal;
	double wolfe1;
	double wolfe2;
	bool   vbsFlg;

	double *d_apHMat;   // optVarNum * optVarNum
	double *d_dirVec;   // optVarNum
	double *d_posNxt;   // optVarNum
	double *d_grdNxt;   // optVarNum
	double *d_dspVec;   // optVarNum
	double *d_dgdVec;   // optVarNum
	double *d_tmpVec;   // optVarNum
};

#endif
