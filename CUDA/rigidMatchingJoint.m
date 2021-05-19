function [h_rgdNdeStk, h_optCtlStk, h_optObjVal, h_optGrdStk] = ...
   rigidMatchingJoint(optObj, h_rgdNdeIniMat, h_wgtGrpVec, h_cntGrpVec, h_cenGrpMat, ...
                      elmObj, tgtObj, knlOrder, knlWidth, knlEps, timeStp)

% --------------- %
%  Preprocessing  %
% --------------- %

dimNum = size(h_rgdNdeIniMat, 1);

endTime = 1;
timeNum = floor(endTime / timeStp) + 1;

rgdGrpNum   = length(h_cntGrpVec);
h_grpNdeVec = repelem(1 : rgdGrpNum, h_cntGrpVec);

h_grpIfoMat = zeros(1 + max(h_cntGrpVec), rgdGrpNum);
for rgdGrpIdx = 1 : rgdGrpNum

	ndeIdxVec = find(h_grpNdeVec == rgdGrpIdx);
	grpCntNum = length(ndeIdxVec);
	h_grpIfoMat(1 : (1 + grpCntNum), rgdGrpIdx) = [grpCntNum; ndeIdxVec(:) - 1];

end

d_rgdNdeIniMat = gpuArray(h_rgdNdeIniMat');
d_grpNdeVec    = gpuArray(int32(h_grpNdeVec - 1));
d_grpIfoMat    = gpuArray(int32(h_grpIfoMat'   ));
d_wgtGrpVec    = gpuArray(h_wgtGrpVec);

d_cenGrpMat = gpuArray(h_cenGrpMat');
d_cenIniMat = d_cenGrpMat(h_grpNdeVec, :);
d_difIniMat = d_rgdNdeIniMat - d_cenIniMat;

if strcmpi(tgtObj.type, 'l2')
	
	tgtObj.tgtNdeMat = gpuArray(tgtObj.tgtNdeMat');

elseif strcmpi(tgtObj.type, 'varifold')

	d_vfdElmVtxMat = elmObj.vfdElmVtxMat;
	d_vfdElmIfoMat = elmObj.vfdElmIfoMat;

	tgtObj.cenPosMat = gpuArray(tgtObj.cenPosMat'  );
	tgtObj.uniDirMat = gpuArray(tgtObj.uniDirMat'  );
	tgtObj.elmVolVec = gpuArray(tgtObj.elmVolVec(:));

else

	error('Unknown target type %s.', tgtObj.type);

end


% -------------- %
%  Optimization  %
% -------------- %

if dimNum == 2

	d_optIniStk = zeros(rgdGrpNum, 3, timeNum - 1, 'gpuArray');

	if strcmpi(tgtObj.type, 'l2')

		[h_optObjVal, d_optCtlStk, d_optGrdStk] = ...
		   rigidMatching_2D_l2(d_optIniStk, optObj.MaxIterations, optObj.OptimalityTolerance, ...
		                       optObj.Wolfe1, optObj.Wolfe2, optObj.Verbose, ...
		                       d_cenIniMat, d_difIniMat, d_grpNdeVec, d_grpIfoMat, d_wgtGrpVec, ...
		                       tgtObj.tgtNdeMat, knlOrder, knlWidth, knlEps, ...
		                       timeStp, timeNum, tgtObj.tgtWgt);

	else

		[h_optObjVal, d_optCtlStk, d_optGrdStk] = ...
		   rigidMatching_2D_varifold(d_optIniStk, optObj.MaxIterations, optObj.OptimalityTolerance, ...
		                             optObj.Wolfe1, optObj.Wolfe2, optObj.Verbose, ...
		                             d_cenIniMat, d_difIniMat, d_grpNdeVec, d_grpIfoMat, d_wgtGrpVec, ...
		                             d_vfdElmVtxMat, d_vfdElmIfoMat, ...
			                          tgtObj.cenPosMat, tgtObj.uniDirMat, tgtObj.elmVolVec, ...
			                          tgtObj.cenKnlType, tgtObj.cenKnlWidth, tgtObj.dirKnlType, tgtObj.dirKnlWidth, ...
		                             knlOrder, knlWidth, knlEps, ...
		                             timeStp, timeNum, tgtObj.tgtWgt);

	end

else

	d_optIniStk = zeros(rgdGrpNum, 6, timeNum - 1, 'gpuArray');

	if strcmpi(tgtObj.type, 'l2')

		[h_optObjVal, d_optCtlStk, d_optGrdStk] = ...
		   rigidMatching_3D_l2(d_optIniStk, optObj.MaxIterations, optObj.OptimalityTolerance, ...
		                       optObj.Wolfe1, optObj.Wolfe2, optObj.Verbose, ...
		                       d_cenIniMat, d_difIniMat, d_grpNdeVec, d_grpIfoMat, d_wgtGrpVec, ...
		                       tgtObj.tgtNdeMat, knlOrder, knlWidth, knlEps, ...
		                       timeStp, timeNum, tgtObj.tgtWgt);
	
	end

end


% ----------------- %
%  Post processing  %
% ----------------- %

h_rgdNdeStk = deformationJoint(d_optCtlStk, d_rgdNdeIniMat, d_cenIniMat, elmObj, h_grpNdeVec, d_wgtGrpVec, ...
                               tgtObj, knlOrder, knlWidth, knlEps, timeStp, timeNum);

h_optCtlStk = permute(gather(d_optCtlStk), [2, 1, 3]);
h_optGrdStk = permute(gather(d_optGrdStk), [2, 1, 3]);

