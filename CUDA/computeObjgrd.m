function [h_objVal, d_grdVec] = ...
   computeObjgrd(d_ctlGrpVec, d_rgdNdeIni, elmObj, h_grpNdeVec, d_wgtGrpVec, ...
                 tgtObj, knlOrder, knlWidth, knlEps, timeStp, timeNum)

dimNum = size(d_rgdNdeIni, 2);

rgdGrpNum   = length(d_wgtGrpVec);
grpCntVec   = histcounts(h_grpNdeVec, 1 : (rgdGrpNum + 1));
h_grpIfoMat = zeros(1 + max(grpCntVec), rgdGrpNum);
for rgdGrpIdx = 1 : rgdGrpNum

	ndeIdxVec = find(h_grpNdeVec == rgdGrpIdx);
	grpCntNum = length(ndeIdxVec);
	h_grpIfoMat(1 : (1 + grpCntNum), rgdGrpIdx) = [grpCntNum; ndeIdxVec(:) - 1];

end

d_grpNdeVec = gpuArray(int32(h_grpNdeVec - 1));
d_grpIfoMat = gpuArray(int32(h_grpIfoMat'   ));

d_cenGrpMat = zeros(rgdGrpNum, dimNum, 'gpuArray');
for rgdGrpIdx = 1 : rgdGrpNum
	d_cenGrpMat(rgdGrpIdx, :) = mean(d_rgdNdeIni(h_grpNdeVec == rgdGrpIdx, :));
end
d_cenIniMat = d_cenGrpMat(h_grpNdeVec, :);
d_difIniMat = d_rgdNdeIni - d_cenIniMat;

if dimNum == 2

	if strcmpi(tgtObj.type, 'l2')

		[h_objVal, d_grdVec] = ...
		   computeObjgrd_2D_l2(d_ctlGrpVec, d_cenIniMat, d_difIniMat, d_grpNdeVec, d_grpIfoMat, d_wgtGrpVec, ...
		                       tgtObj.tgtNdeMat, knlOrder, knlWidth, knlEps, ...
		                       timeStp, timeNum, tgtObj.tgtWgt);

	elseif strcmpi(tgtObj.type, 'varifold')

		d_vfdElmVtxMat = elmObj.vfdElmVtxMat;
		d_vfdElmIfoMat = elmObj.vfdElmIfoMat;

		[h_objVal, d_grdVec] = ...
		   computeObjgrd_2D_varifold(d_ctlGrpVec, d_cenIniMat, d_difIniMat, d_grpNdeVec, d_grpIfoMat, d_wgtGrpVec, ...
		                             d_vfdElmVtxMat, d_vfdElmIfoMat, ...
		                             tgtObj.cenPosMat, tgtObj.uniDirMat, tgtObj.elmVolVec, ...
		                             tgtObj.cenKnlType, tgtObj.cenKnlWidth, tgtObj.dirKnlType, tgtObj.dirKnlWidth, ...
		                             knlOrder, knlWidth, knlEps, timeStp, timeNum, tgtObj.tgtWgt);

	end

else

	if strcmpi(tgtObj.type, 'l2')

		[h_objVal, d_grdVec] = ...
		   computeObjgrd_3D_l2(d_ctlGrpVec, d_cenIniMat, d_difIniMat, d_grpNdeVec, d_grpIfoMat, d_wgtGrpVec, ...
		                       tgtObj.tgtNdeMat, knlOrder, knlWidth, knlEps, ...
		                       timeStp, timeNum, tgtObj.tgtWgt);

	end

end
