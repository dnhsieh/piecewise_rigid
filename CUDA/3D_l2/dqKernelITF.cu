#include "struct.h"

void dqKernel(double *, double *, double *, double *, int, double, int);

void dqKernel(double *d_dqKMat, double *d_rgdNdeMat,
              double *d_lftMat, double *d_rgtMat, fcndata &fcnObj)
{
	int    knlOrder  = fcnObj.prm.knlOrder;
	double knlWidth  = fcnObj.prm.knlWidth;
	int    rgdNdeNum = fcnObj.prm.rgdNdeNum;

	dqKernel(d_dqKMat, d_rgdNdeMat, d_lftMat, d_rgtMat, knlOrder, knlWidth, rgdNdeNum);

	return;
}

