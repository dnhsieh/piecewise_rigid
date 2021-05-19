#include "struct.h"

void computeKernel(double *, double *, int, double, int);

void computeKernel(double *d_knlMat, double *d_rgdNdeMat, fcndata &fcnObj)
{
	int    knlOrder  = fcnObj.prm.knlOrder;
	double knlWidth  = fcnObj.prm.knlWidth;
	int    rgdNdeNum = fcnObj.prm.rgdNdeNum;

	computeKernel(d_knlMat, d_rgdNdeMat, knlOrder, knlWidth, rgdNdeNum);

	return;
}

