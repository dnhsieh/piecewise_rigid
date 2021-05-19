knlOrder = 3;
knlWidth = 0.06;
knlEps   = 1e-6;
tgtWgt   = 500;
timeStp  = 0.005;

cenKnlType  = 'C';
cenKnlWidth = 0.2;
dirKnlType  = 'B';
dirKnlWidth = 0.05;

optObj.MaxIterations       = 100000;
optObj.OptimalityTolerance = 1e-6;
optObj.Wolfe1              = 0;
optObj.Wolfe2              = 0.5;
optObj.Verbose             = true;

load initialShape allNdeIni cntGrpVec cenGrpMat rgdElmVtxMat
load targetShape tgtObj

tgtObj.type        = 'varifold';
tgtObj.tgtWgt      = tgtWgt;
tgtObj.cenKnlType  = cenKnlType;
tgtObj.cenKnlWidth = cenKnlWidth;
tgtObj.dirKnlType  = dirKnlType;
tgtObj.dirKnlWidth = dirKnlWidth;

addpath ../../CUDA

wgtGrpVec = ones(size(cntGrpVec));

rgdNdeNum = sum(cntGrpVec);
rgdNdeIni = allNdeIni(:, 1 : rgdNdeNum);
elmObj    = generateElementObject(rgdNdeIni, rgdElmVtxMat);

[allNdeStk, ctlStk, objVal, grdStk] = ...
   rigidMatchingJoint(optObj, rgdNdeIni, wgtGrpVec, cntGrpVec, cenGrpMat, elmObj, tgtObj, ...
                      knlOrder, knlWidth, knlEps, timeStp);

rmpath ../../CUDA
