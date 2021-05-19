function elmObj = generateElementObject(rgdNdeMat, rgdElmVtxMat)

rgdNdeNum = size(rgdNdeMat, 2);

triObj       = triangulation(rgdElmVtxMat', rgdNdeMat');
vfdElmVtxMat = triObj.freeBoundary';

vfdAdjCntVec = histcounts(vfdElmVtxMat(:), 1 : (rgdNdeNum + 1));
vfdElmIfoMat = zeros(1 + 2 * max(vfdAdjCntVec), rgdNdeNum);

for rgdNdeIdx = 1 : rgdNdeNum

	[lclIdxVec, elmIdxVec] = find(vfdElmVtxMat == rgdNdeIdx);

	adjNum = length(elmIdxVec);
	vfdElmIfoMat(1, rgdNdeIdx) = adjNum;

	for adjIdx = 1 : adjNum
		if lclIdxVec(adjIdx) == 1
			vfdElmIfoMat(1 + 2 * (adjIdx - 1) + (1 : 2), rgdNdeIdx) = [elmIdxVec(adjIdx) - 1, -1];
		else
			vfdElmIfoMat(1 + 2 * (adjIdx - 1) + (1 : 2), rgdNdeIdx) = [elmIdxVec(adjIdx) - 1,  1];
		end
	end

end

% ---

elmObj.vfdElmVtxMat = gpuArray(int32(vfdElmVtxMat' - 1));
elmObj.vfdElmIfoMat = gpuArray(int32(vfdElmIfoMat'    ));

