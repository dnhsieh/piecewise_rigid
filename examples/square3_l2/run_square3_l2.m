knlOrder = 3;
knlWidth = 0.06;
knlEps   = 1e-6;
tgtWgt   = 100;
timeStp  = 0.005;

optObj.MaxIterations       = 100000;
optObj.OptimalityTolerance = 1e-6;
optObj.Wolfe1              = 0;
optObj.Wolfe2              = 0.5;
optObj.Verbose             = true;

load initialShape allNdeIni cntGrpVec cenGrpMat rgdElmVtxMat
load targetShape tgtObj

tgtObj.type   = 'l2';
tgtObj.tgtWgt = tgtWgt;

addpath ../../CUDA

wgtGrpVec = ones(size(cntGrpVec));

rgdNdeNum = sum(cntGrpVec);
rgdNdeIni = allNdeIni(:, 1 : rgdNdeNum);

[allNdeStk, ctlStk, objVal, grdStk] = ...
   rigidMatchingJoint(optObj, rgdNdeIni, wgtGrpVec, cntGrpVec, cenGrpMat, [], tgtObj, ...
                      knlOrder, knlWidth, knlEps, timeStp);

rmpath ../../CUDA
