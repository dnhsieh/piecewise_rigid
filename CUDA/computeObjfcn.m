function h_objVal = ...
   computeObjfcn(d_ctlGrpVec, d_rgdNdeIni, elmObj, h_grpNdeVec, d_wgtGrpVec, ...
                 tgtObj, knlOrder, knlWidth, knlEps, timeStp, timeNum)

dimNum = size(d_rgdNdeIni, 2);

d_grpNdeVec = gpuArray(int32(h_grpNdeVec - 1));

rgdGrpNum   = length(d_wgtGrpVec);
d_cenGrpMat = zeros(rgdGrpNum, dimNum, 'gpuArray');
for rgdGrpIdx = 1 : rgdGrpNum
	d_cenGrpMat(rgdGrpIdx, :) = mean(d_rgdNdeIni(h_grpNdeVec == rgdGrpIdx, :));
end
d_cenIniMat = d_cenGrpMat(h_grpNdeVec, :);
d_difIniMat = d_rgdNdeIni - d_cenIniMat;

if dimNum == 2

	if strcmpi(tgtObj.type, 'l2')

		h_objVal = ...
		   computeObjfcn_2D_l2(d_ctlGrpVec, d_cenIniMat, d_difIniMat, d_grpNdeVec, d_wgtGrpVec, ...
		                       tgtObj.tgtNdeMat, knlOrder, knlWidth, knlEps, ...
		                       timeStp, timeNum, tgtObj.tgtWgt);
	
	elseif strcmpi(tgtObj.type, 'varifold')

		d_vfdElmVtxMat = elmObj.vfdElmVtxMat;

		h_objVal = ...
		   computeObjfcn_2D_varifold(d_ctlGrpVec, d_cenIniMat, d_difIniMat, d_grpNdeVec, d_wgtGrpVec, ...
		                             d_vfdElmVtxMat, tgtObj.cenPosMat, tgtObj.uniDirMat, tgtObj.elmVolVec, ...
		                             tgtObj.cenKnlType, tgtObj.cenKnlWidth, tgtObj.dirKnlType, tgtObj.dirKnlWidth, ...
		                             knlOrder, knlWidth, knlEps, timeStp, timeNum, tgtObj.tgtWgt);
	
	end

else

	if strcmpi(tgtObj.type, 'l2')

		h_objVal = ...
		   computeObjfcn_3D_l2(d_ctlGrpVec, d_cenIniMat, d_difIniMat, d_grpNdeVec, d_wgtGrpVec, ...
		                       tgtObj.tgtNdeMat, knlOrder, knlWidth, knlEps, ...
		                       timeStp, timeNum, tgtObj.tgtWgt);
	
	end

end
