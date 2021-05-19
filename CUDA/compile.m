function compile(fcnStr)

if nargin == 0
	fcnStr = 'all';
end

includeDir = 'include';
libraryDir = '/usr/local/cuda/lib64';
matlabFlag = '-R2018a';

% - - -

if   strcmpi(fcnStr, '2D_objfcn_varifold') || strcmpi(fcnStr, '2D_objfcn') ...
  || strcmpi(fcnStr, '2D_varifold') || strcmpi(fcnStr, '2D') || strcmpi(fcnStr, 'all')

	files = ...
	{
	'2D_varifold/computeObjfcn.cu', ...
	'2D_varifold/assignStructMemory.cu', ...
	'2D_varifold/objfcn.cu', ...
	'2D_varifold/group2Node.cu', ...
	'2D_varifold/computeControlCost.cu', ...
	'2D_varifold/computeKernelITF.cu', ...
	'2D_varifold/computeKernel.cu', ...
	'2D_varifold/addEpsIdentity.cu', ...
	'2D_varifold/cholesky.cu', ...
	'2D_varifold/computeRigidVelocity.cu', ...
	'2D_varifold/computeRigidDeformation.cu', ...
	'2D_varifold/varifoldITF.cu', ...
	'2D_varifold/varifold.cu', ...
	'2D_varifold/dsum.cu', ...
	'2D_varifold/vectorSum.cu', ...
	};
	
	fprintf('Compiling computeObjfcn (2D, varifold)...\n');
	mexcuda(matlabFlag, '-DDIM2', ['-L', libraryDir], ['-I', includeDir], files, ...
	        '-lcublas', '-lcusolver', '-output', 'computeObjfcn_2D_varifold');
	fprintf('\n');

end

if   strcmpi(fcnStr, '2D_objgrd_varifold') || strcmpi(fcnStr, '2D_objgrd') ...
  || strcmpi(fcnStr, '2D_varifold') || strcmpi(fcnStr, '2D') || strcmpi(fcnStr, 'all')

	files = ...
	{
	'2D_varifold/computeObjgrd.cu', ...
	'2D_varifold/assignStructMemory.cu', ...
	'2D_varifold/objgrd.cu', ...
	'2D_varifold/group2Node.cu', ...
	'2D_varifold/computeControlCost.cu', ...
	'2D_varifold/computeKernelITF.cu', ...
	'2D_varifold/computeKernel.cu', ...
	'2D_varifold/addEpsIdentity.cu', ...
	'2D_varifold/cholesky.cu', ...
	'2D_varifold/computeRigidVelocity.cu', ...
	'2D_varifold/computeRigidDeformation.cu', ...
	'2D_varifold/varifoldITF.cu', ...
	'2D_varifold/varifold.cu', ...
	'2D_varifold/duRigidVelocity.cu', ...
	'2D_varifold/dqKernelITF.cu', ...
	'2D_varifold/dqKernel.cu', ...
	'2D_varifold/dqRigidVelocity.cu', ...
	'2D_varifold/dqRigidDeformation.cu', ...
	'2D_varifold/duControl.cu', ...
	'2D_varifold/node2Group.cu', ...
	'2D_varifold/dsum.cu', ...
	'2D_varifold/vectorSum.cu', ...
	'2D_varifold/vectorScale.cu', ...
	};
	
	fprintf('Compiling computeObjgrd (2D, varifold)...\n');
	mexcuda(matlabFlag, '-DDIM2', ['-L', libraryDir], ['-I', includeDir], files, ...
	        '-lcublas', '-lcusolver', '-output', 'computeObjgrd_2D_varifold');
	fprintf('\n');

end

if   strcmpi(fcnStr, '2D_deformation_varifold') || strcmpi(fcnStr, '2D_deformation') ...
  || strcmpi(fcnStr, '2D_varifold') || strcmpi(fcnStr, '2D') || strcmpi(fcnStr, 'all')

	files = ...
	{
	'2D_varifold/deformation.cu', ...
	'2D_varifold/assignStructMemory.cu', ...
	'2D_varifold/objfcn.cu', ...
	'2D_varifold/group2Node.cu', ...
	'2D_varifold/computeControlCost.cu', ...
	'2D_varifold/computeKernelITF.cu', ...
	'2D_varifold/computeKernel.cu', ...
	'2D_varifold/addEpsIdentity.cu', ...
	'2D_varifold/cholesky.cu', ...
	'2D_varifold/computeRigidVelocity.cu', ...
	'2D_varifold/computeRigidDeformation.cu', ...
	'2D_varifold/varifoldITF.cu', ...
	'2D_varifold/varifold.cu', ...
	'2D_varifold/dsum.cu', ...
	'2D_varifold/vectorSum.cu', ...
	};
	
	fprintf('Compiling deformation (2D, varifold)...\n');
	mexcuda(matlabFlag, '-DDIM2', ['-L', libraryDir], ['-I', includeDir], files, ...
	        '-lcublas', '-lcusolver', '-output', 'deformation_2D_varifold');
	fprintf('\n');

end

if   strcmpi(fcnStr, '2D_matching_varifold') || strcmpi(fcnStr, '2D_matching') ...
  || strcmpi(fcnStr, '2D_varifold') || strcmpi(fcnStr, '2D') || strcmpi(fcnStr, 'all')

	files = ...
	{
	'2D_varifold/rigidMatching.cu', ...
	'2D_varifold/assignStructMemory.cu', ...
	'2D_varifold/BFGS.cu', ...
	'2D_varifold/lineSearch.cu', ...
	'2D_varifold/updateInverseHessian.cu', ...
	'2D_varifold/objgrd.cu', ...
	'2D_varifold/group2Node.cu', ...
	'2D_varifold/computeControlCost.cu', ...
	'2D_varifold/computeKernelITF.cu', ...
	'2D_varifold/computeKernel.cu', ...
	'2D_varifold/addEpsIdentity.cu', ...
	'2D_varifold/cholesky.cu', ...
	'2D_varifold/computeRigidVelocity.cu', ...
	'2D_varifold/computeRigidDeformation.cu', ...
	'2D_varifold/varifoldITF.cu', ...
	'2D_varifold/varifold.cu', ...
	'2D_varifold/duRigidVelocity.cu', ...
	'2D_varifold/dqKernelITF.cu', ...
	'2D_varifold/dqKernel.cu', ...
	'2D_varifold/dqRigidVelocity.cu', ...
	'2D_varifold/dqRigidDeformation.cu', ...
	'2D_varifold/duControl.cu', ...
	'2D_varifold/node2Group.cu', ...
	'2D_varifold/dsum.cu', ...
	'2D_varifold/vectorSum.cu', ...
	'2D_varifold/vectorScale.cu', ...
	};
	
	fprintf('Compiling rigidMatching (2D, varifold)...\n');
	mexcuda(matlabFlag, '-DDIM2', ['-L', libraryDir], ['-I', includeDir], files, ...
	        '-lcublas', '-lcusolver', '-output', 'rigidMatching_2D_varifold');
	fprintf('\n');

end

% - - -

if   strcmpi(fcnStr, '2D_objfcn_l2') || strcmpi(fcnStr, '2D_objfcn') ...
  || strcmpi(fcnStr, '2D_l2') || strcmpi(fcnStr, '2D') || strcmpi(fcnStr, 'all')

	files = ...
	{
	'2D_l2/computeObjfcn.cu', ...
	'2D_l2/assignStructMemory.cu', ...
	'2D_l2/objfcn.cu', ...
	'2D_l2/group2Node.cu', ...
	'2D_l2/computeControlCost.cu', ...
	'2D_l2/computeKernelITF.cu', ...
	'2D_l2/computeKernel.cu', ...
	'2D_l2/addEpsIdentity.cu', ...
	'2D_l2/cholesky.cu', ...
	'2D_l2/computeRigidVelocity.cu', ...
	'2D_l2/computeRigidDeformation.cu', ...
	'2D_l2/dsum.cu', ...
	'2D_l2/vectorSum.cu', ...
	};
	
	fprintf('Compiling computeObjfcn (2D, l2)...\n');
	mexcuda(matlabFlag, '-DDIM2', ['-L', libraryDir], ['-I', includeDir], files, ...
	        '-lcublas', '-lcusolver', '-output', 'computeObjfcn_2D_l2');
	fprintf('\n');

end

if   strcmpi(fcnStr, '2D_objgrd_l2') || strcmpi(fcnStr, '2D_objgrd') ...
  || strcmpi(fcnStr, '2D_l2') || strcmpi(fcnStr, '2D') || strcmpi(fcnStr, 'all')

	files = ...
	{
	'2D_l2/computeObjgrd.cu', ...
	'2D_l2/assignStructMemory.cu', ...
	'2D_l2/objgrd.cu', ...
	'2D_l2/group2Node.cu', ...
	'2D_l2/computeControlCost.cu', ...
	'2D_l2/computeKernelITF.cu', ...
	'2D_l2/computeKernel.cu', ...
	'2D_l2/addEpsIdentity.cu', ...
	'2D_l2/cholesky.cu', ...
	'2D_l2/computeRigidVelocity.cu', ...
	'2D_l2/computeRigidDeformation.cu', ...
	'2D_l2/duRigidVelocity.cu', ...
	'2D_l2/dqKernelITF.cu', ...
	'2D_l2/dqKernel.cu', ...
	'2D_l2/dqRigidVelocity.cu', ...
	'2D_l2/dqRigidDeformation.cu', ...
	'2D_l2/duControl.cu', ...
	'2D_l2/node2Group.cu', ...
	'2D_l2/dsum.cu', ...
	'2D_l2/vectorSum.cu', ...
	'2D_l2/vectorScale.cu', ...
	};
	
	fprintf('Compiling computeObjgrd (2D, l2)...\n');
	mexcuda(matlabFlag, '-DDIM2', ['-L', libraryDir], ['-I', includeDir], files, ...
	        '-lcublas', '-lcusolver', '-output', 'computeObjgrd_2D_l2');
	fprintf('\n');

end

if   strcmpi(fcnStr, '2D_deformation_l2') || strcmpi(fcnStr, '2D_deformation') ...
  || strcmpi(fcnStr, '2D_l2') || strcmpi(fcnStr, '2D') || strcmpi(fcnStr, 'all')

	files = ...
	{
	'2D_l2/deformation.cu', ...
	'2D_l2/assignStructMemory.cu', ...
	'2D_l2/objfcn.cu', ...
	'2D_l2/group2Node.cu', ...
	'2D_l2/computeControlCost.cu', ...
	'2D_l2/computeKernelITF.cu', ...
	'2D_l2/computeKernel.cu', ...
	'2D_l2/addEpsIdentity.cu', ...
	'2D_l2/cholesky.cu', ...
	'2D_l2/computeRigidVelocity.cu', ...
	'2D_l2/computeRigidDeformation.cu', ...
	'2D_l2/dsum.cu', ...
	'2D_l2/vectorSum.cu', ...
	};
	
	fprintf('Compiling deformation (2D, l2)...\n');
	mexcuda(matlabFlag, '-DDIM2', ['-L', libraryDir], ['-I', includeDir], files, ...
	        '-lcublas', '-lcusolver', '-output', 'deformation_2D_l2');
	fprintf('\n');

end

if   strcmpi(fcnStr, '2D_matching_l2') || strcmpi(fcnStr, '2D_matching') ...
  || strcmpi(fcnStr, '2D_l2') || strcmpi(fcnStr, '2D') || strcmpi(fcnStr, 'all')

	files = ...
	{
	'2D_l2/rigidMatching.cu', ...
	'2D_l2/assignStructMemory.cu', ...
	'2D_l2/BFGS.cu', ...
	'2D_l2/lineSearch.cu', ...
	'2D_l2/updateInverseHessian.cu', ...
	'2D_l2/objgrd.cu', ...
	'2D_l2/group2Node.cu', ...
	'2D_l2/computeControlCost.cu', ...
	'2D_l2/computeKernelITF.cu', ...
	'2D_l2/computeKernel.cu', ...
	'2D_l2/addEpsIdentity.cu', ...
	'2D_l2/cholesky.cu', ...
	'2D_l2/computeRigidVelocity.cu', ...
	'2D_l2/computeRigidDeformation.cu', ...
	'2D_l2/duRigidVelocity.cu', ...
	'2D_l2/dqKernelITF.cu', ...
	'2D_l2/dqKernel.cu', ...
	'2D_l2/dqRigidVelocity.cu', ...
	'2D_l2/dqRigidDeformation.cu', ...
	'2D_l2/duControl.cu', ...
	'2D_l2/node2Group.cu', ...
	'2D_l2/dsum.cu', ...
	'2D_l2/vectorSum.cu', ...
	'2D_l2/vectorScale.cu', ...
	};
	
	fprintf('Compiling rigidMatching (2D, l2)...\n');
	mexcuda(matlabFlag, '-DDIM2', ['-L', libraryDir], ['-I', includeDir], files, ...
	        '-lcublas', '-lcusolver', '-output', 'rigidMatching_2D_l2');
	fprintf('\n');

end

% - - -

if   strcmpi(fcnStr, '3D_objfcn_l2') || strcmpi(fcnStr, '3D_objfcn') ...
  || strcmpi(fcnStr, '3D_l2') || strcmpi(fcnStr, '3D') || strcmpi(fcnStr, 'all')

	files = ...
	{
	'3D_l2/computeObjfcn.cu', ...
	'3D_l2/assignStructMemory.cu', ...
	'3D_l2/objfcn.cu', ...
	'3D_l2/group2Node.cu', ...
	'3D_l2/computeControlCost.cu', ...
	'3D_l2/computeKernelITF.cu', ...
	'3D_l2/computeKernel.cu', ...
	'3D_l2/addEpsIdentity.cu', ...
	'3D_l2/cholesky.cu', ...
	'3D_l2/computeRigidVelocity.cu', ...
	'3D_l2/computeRigidDeformation.cu', ...
	'3D_l2/dsum.cu', ...
	'3D_l2/vectorSum.cu', ...
	};
	
	fprintf('Compiling computeObjfcn (3D, l2)...\n');
	mexcuda(matlabFlag, '-DDIM3', ['-L', libraryDir], ['-I', includeDir], files, ...
	        '-lcublas', '-lcusolver', '-output', 'computeObjfcn_3D_l2');
	fprintf('\n');

end

if   strcmpi(fcnStr, '3D_objgrd_l2') || strcmpi(fcnStr, '3D_objgrd') ...
  || strcmpi(fcnStr, '3D_l2') || strcmpi(fcnStr, '3D') || strcmpi(fcnStr, 'all')

	files = ...
	{
	'3D_l2/computeObjgrd.cu', ...
	'3D_l2/assignStructMemory.cu', ...
	'3D_l2/objgrd.cu', ...
	'3D_l2/group2Node.cu', ...
	'3D_l2/computeControlCost.cu', ...
	'3D_l2/computeKernelITF.cu', ...
	'3D_l2/computeKernel.cu', ...
	'3D_l2/addEpsIdentity.cu', ...
	'3D_l2/cholesky.cu', ...
	'3D_l2/computeRigidVelocity.cu', ...
	'3D_l2/computeRigidDeformation.cu', ...
	'3D_l2/duRigidVelocity.cu', ...
	'3D_l2/dqKernelITF.cu', ...
	'3D_l2/dqKernel.cu', ...
	'3D_l2/dqRigidVelocity.cu', ...
	'3D_l2/dqRigidDeformation.cu', ...
	'3D_l2/duControl.cu', ...
	'3D_l2/node2Group.cu', ...
	'3D_l2/dsum.cu', ...
	'3D_l2/vectorSum.cu', ...
	'3D_l2/vectorScale.cu', ...
	};
	
	fprintf('Compiling computeObjgrd (3D, l2)...\n');
	mexcuda(matlabFlag, '-DDIM3', ['-L', libraryDir], ['-I', includeDir], files, ...
	        '-lcublas', '-lcusolver', '-output', 'computeObjgrd_3D_l2');
	fprintf('\n');

end

if   strcmpi(fcnStr, '3D_deformation_l2') || strcmpi(fcnStr, '3D_deformation') ...
  || strcmpi(fcnStr, '3D_l2') || strcmpi(fcnStr, '3D') || strcmpi(fcnStr, 'all')

	files = ...
	{
	'3D_l2/deformation.cu', ...
	'3D_l2/assignStructMemory.cu', ...
	'3D_l2/objfcn.cu', ...
	'3D_l2/group2Node.cu', ...
	'3D_l2/computeControlCost.cu', ...
	'3D_l2/computeKernelITF.cu', ...
	'3D_l2/computeKernel.cu', ...
	'3D_l2/addEpsIdentity.cu', ...
	'3D_l2/cholesky.cu', ...
	'3D_l2/computeRigidVelocity.cu', ...
	'3D_l2/computeRigidDeformation.cu', ...
	'3D_l2/dsum.cu', ...
	'3D_l2/vectorSum.cu', ...
	};
	
	fprintf('Compiling deformation (3D, l2)...\n');
	mexcuda(matlabFlag, '-DDIM3', ['-L', libraryDir], ['-I', includeDir], files, ...
	        '-lcublas', '-lcusolver', '-output', 'deformation_3D_l2');
	fprintf('\n');

end

if   strcmpi(fcnStr, '3D_matching_l2') || strcmpi(fcnStr, '3D_matching') ...
  || strcmpi(fcnStr, '3D_l2') || strcmpi(fcnStr, '3D') || strcmpi(fcnStr, 'all')

	files = ...
	{
	'3D_l2/rigidMatching.cu', ...
	'3D_l2/assignStructMemory.cu', ...
	'3D_l2/BFGS.cu', ...
	'3D_l2/lineSearch.cu', ...
	'3D_l2/updateInverseHessian.cu', ...
	'3D_l2/objgrd.cu', ...
	'3D_l2/group2Node.cu', ...
	'3D_l2/computeControlCost.cu', ...
	'3D_l2/computeKernelITF.cu', ...
	'3D_l2/computeKernel.cu', ...
	'3D_l2/addEpsIdentity.cu', ...
	'3D_l2/cholesky.cu', ...
	'3D_l2/computeRigidVelocity.cu', ...
	'3D_l2/computeRigidDeformation.cu', ...
	'3D_l2/duRigidVelocity.cu', ...
	'3D_l2/dqKernelITF.cu', ...
	'3D_l2/dqKernel.cu', ...
	'3D_l2/dqRigidVelocity.cu', ...
	'3D_l2/dqRigidDeformation.cu', ...
	'3D_l2/duControl.cu', ...
	'3D_l2/node2Group.cu', ...
	'3D_l2/dsum.cu', ...
	'3D_l2/vectorSum.cu', ...
	'3D_l2/vectorScale.cu', ...
	};
	
	fprintf('Compiling rigidMatching (3D, l2)...\n');
	mexcuda(matlabFlag, '-DDIM3', ['-L', libraryDir], ['-I', includeDir], files, ...
	        '-lcublas', '-lcusolver', '-output', 'rigidMatching_3D_l2');
	fprintf('\n');

end

