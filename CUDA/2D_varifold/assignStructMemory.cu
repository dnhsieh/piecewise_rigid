#include "constants.h"
#include "struct.h"

void assignObjfcnStructMemory(long long &memCnt, fcndata &fcnObj, double *d_fcnWorkspace)
{
	int rgdGrpNum = fcnObj.prm.rgdGrpNum;
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;
	int vfdElmNum = fcnObj.prm.vfdElmNum;
	int tgtElmNum = fcnObj.tgt.tgtElmNum;
	int   timeNum = fcnObj.prm.timeNum;

	// ---

	memCnt = 0;

	double *d_nowPtr = d_fcnWorkspace;

	// ---

	fcnObj.d_ctlNdeStk = d_nowPtr;
	d_nowPtr += rgdNdeNum * RGDDOF * (timeNum - 1);
	memCnt   += rgdNdeNum * RGDDOF * (timeNum - 1);

	fcnObj.d_ctlCumStk = d_nowPtr;
	d_nowPtr += rgdNdeNum * RGDDOF * timeNum;
	memCnt   += rgdNdeNum * RGDDOF * timeNum;

	fcnObj.d_rgdNdeStk = d_nowPtr;
	d_nowPtr += rgdNdeNum * DIMNUM * timeNum;
	memCnt   += rgdNdeNum * DIMNUM * timeNum;

	fcnObj.d_ctlVec = d_nowPtr;
	d_nowPtr += rgdGrpNum * (timeNum - 1);
	memCnt   += rgdGrpNum * (timeNum - 1);

	fcnObj.d_rgdVlcStk = d_nowPtr;
	d_nowPtr += rgdNdeNum * DIMNUM * (timeNum - 1);
	memCnt   += rgdNdeNum * DIMNUM * (timeNum - 1);

	fcnObj.d_rgdAlpStk = d_nowPtr;
	d_nowPtr += rgdNdeNum * DIMNUM * (timeNum - 1);
	memCnt   += rgdNdeNum * DIMNUM * (timeNum - 1);

	fcnObj.d_rgdKnlMat = d_nowPtr;
	d_nowPtr += rgdNdeNum * rgdNdeNum;
	memCnt   += rgdNdeNum * rgdNdeNum;

	fcnObj.d_rgdKnLMat = d_nowPtr;
	d_nowPtr += rgdNdeNum * rgdNdeNum;
	memCnt   += rgdNdeNum * rgdNdeNum;

	fcnObj.d_dfmCenPosMat = d_nowPtr;
	d_nowPtr += vfdElmNum * DIMNUM;
	memCnt   += vfdElmNum * DIMNUM;

	fcnObj.d_dfmUniDirMat = d_nowPtr;
	d_nowPtr += vfdElmNum * DIMNUM;
	memCnt   += vfdElmNum * DIMNUM;

	fcnObj.d_dfmElmVolVec = d_nowPtr;
	d_nowPtr += vfdElmNum;
	memCnt   += vfdElmNum;

	fcnObj.d_vfdVec = d_nowPtr;
	d_nowPtr += vfdElmNum + tgtElmNum;
	memCnt   += vfdElmNum + tgtElmNum;

	fcnObj.d_sumBufVec = d_nowPtr;
	memCnt += SUMBLKDIM;

	return;
}

void assignObjgrdStructMemory(long long &memCnt, fcndata &fcnObj, double *d_fcnWorkspace)
{
	int rgdGrpNum = fcnObj.prm.rgdGrpNum;
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;
	int vfdElmNum = fcnObj.prm.vfdElmNum;
	int tgtElmNum = fcnObj.tgt.tgtElmNum;
	int   timeNum = fcnObj.prm.timeNum;

	// ---

	memCnt = 0;

	double *d_nowPtr = d_fcnWorkspace;

	// ---

	fcnObj.d_ctlNdeStk = d_nowPtr;
	d_nowPtr += rgdNdeNum * RGDDOF * (timeNum - 1);
	memCnt   += rgdNdeNum * RGDDOF * (timeNum - 1);

	fcnObj.d_ctlCumStk = d_nowPtr;
	d_nowPtr += rgdNdeNum * RGDDOF * timeNum;
	memCnt   += rgdNdeNum * RGDDOF * timeNum;

	fcnObj.d_rgdNdeStk = d_nowPtr;
	d_nowPtr += rgdNdeNum * DIMNUM * timeNum;
	memCnt   += rgdNdeNum * DIMNUM * timeNum;

	fcnObj.d_ctlVec = d_nowPtr;
	d_nowPtr += rgdGrpNum * (timeNum - 1);
	memCnt   += rgdGrpNum * (timeNum - 1);

	fcnObj.d_rgdVlcStk = d_nowPtr;
	d_nowPtr += rgdNdeNum * DIMNUM * (timeNum - 1);
	memCnt   += rgdNdeNum * DIMNUM * (timeNum - 1);

	fcnObj.d_rgdAlpStk = d_nowPtr;
	d_nowPtr += rgdNdeNum * DIMNUM * (timeNum - 1);
	memCnt   += rgdNdeNum * DIMNUM * (timeNum - 1);

	fcnObj.d_rgdKnlMat = d_nowPtr;
	d_nowPtr += rgdNdeNum * rgdNdeNum;
	memCnt   += rgdNdeNum * rgdNdeNum;

	fcnObj.d_rgdKnLMat = d_nowPtr;
	d_nowPtr += rgdNdeNum * rgdNdeNum;
	memCnt   += rgdNdeNum * rgdNdeNum;

	fcnObj.d_dfmCenPosMat = d_nowPtr;
	d_nowPtr += vfdElmNum * DIMNUM;
	memCnt   += vfdElmNum * DIMNUM;

	fcnObj.d_dfmUniDirMat = d_nowPtr;
	d_nowPtr += vfdElmNum * DIMNUM;
	memCnt   += vfdElmNum * DIMNUM;

	fcnObj.d_dfmElmVolVec = d_nowPtr;
	d_nowPtr += vfdElmNum;
	memCnt   += vfdElmNum;

	fcnObj.d_vfdVec = d_nowPtr;
	d_nowPtr += vfdElmNum + tgtElmNum;
	memCnt   += vfdElmNum + tgtElmNum;

	fcnObj.d_dqVfdMat = d_nowPtr;
	d_nowPtr += rgdNdeNum * DIMNUM;
	memCnt   += rgdNdeNum * DIMNUM;

	fcnObj.d_dcVfdMat = d_nowPtr;
	d_nowPtr += vfdElmNum * DIMNUM;
	memCnt   += vfdElmNum * DIMNUM;

	fcnObj.d_ddVfdMat = d_nowPtr;
	d_nowPtr += vfdElmNum * DIMNUM;
	memCnt   += vfdElmNum * DIMNUM;

	fcnObj.d_grdNdeStk = d_nowPtr;
	d_nowPtr += rgdNdeNum * RGDDOF * (timeNum - 1);
	memCnt   += rgdNdeNum * RGDDOF * (timeNum - 1);

	fcnObj.d_pMat = d_nowPtr;
	d_nowPtr += rgdNdeNum * RGDDOF;
	memCnt   += rgdNdeNum * RGDDOF;

	fcnObj.d_pDotMat = d_nowPtr;
	d_nowPtr += rgdNdeNum * RGDDOF;
	memCnt   += rgdNdeNum * RGDDOF;

	fcnObj.d_duRVMat = d_nowPtr;
	d_nowPtr += rgdNdeNum * RGDDOF;
	memCnt   += rgdNdeNum * RGDDOF;

	fcnObj.d_dqKMat = d_nowPtr;
	d_nowPtr += rgdNdeNum * DIMNUM;
	memCnt   += rgdNdeNum * DIMNUM;

	fcnObj.d_dqRVMat = d_nowPtr;
	d_nowPtr += rgdNdeNum * RGDDOF;
	memCnt   += rgdNdeNum * RGDDOF;

	fcnObj.d_dqRDMat = d_nowPtr;
	d_nowPtr += rgdNdeNum * RGDDOF;
	memCnt   += rgdNdeNum * RGDDOF;

	fcnObj.d_sumBufVec = d_nowPtr;
	memCnt += SUMBLKDIM;

	return;
}

void assignOptimizationStructMemory(long long &memCnt, optdata &optObj, double *d_optWorkspace)
{
	int varNum = optObj.varNum;

	// ---

	memCnt = 0;

	double *d_nowPtr = d_optWorkspace;

	// ---

	optObj.d_apHMat = d_nowPtr;
	d_nowPtr += varNum * varNum;
	memCnt   += varNum * varNum;

	optObj.d_dirVec = d_nowPtr;
	d_nowPtr += varNum;
	memCnt   += varNum;

	optObj.d_posNxt = d_nowPtr;
	d_nowPtr += varNum;
	memCnt   += varNum;

	optObj.d_grdNxt = d_nowPtr;
	d_nowPtr += varNum;
	memCnt   += varNum;

	optObj.d_dspVec = d_nowPtr;
	d_nowPtr += varNum;
	memCnt   += varNum;

	optObj.d_dgdVec = d_nowPtr;
	d_nowPtr += varNum;
	memCnt   += varNum;

	optObj.d_tmpVec = d_nowPtr;
	memCnt   += varNum;

	return;
}
