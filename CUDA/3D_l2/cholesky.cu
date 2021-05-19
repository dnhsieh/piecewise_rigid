#include <cusolverDn.h>
#include "struct.h"

void cholesky(double *d_knlMat, fcndata &fcnObj) 
{
	int rgdNdeNum = fcnObj.prm.rgdNdeNum;

	cusolverDnHandle_t solvHdl     = fcnObj.solvHdl;
	int                h_Lwork     = fcnObj.h_Lwork;
	double            *d_workspace = fcnObj.d_workspace;
	int               *d_status    = fcnObj.d_status;

	cusolverDnDpotrf(solvHdl, CUBLAS_FILL_MODE_LOWER, 
	                 rgdNdeNum, d_knlMat, rgdNdeNum, d_workspace, h_Lwork, d_status);

	return;
}
