#ifndef ROTATION_H
#define ROTATION_H

inline __device__ vector applyRotXMat(vector inpVec, vector angVec)
{
	double cosVal = cos(angVec.x);
	double sinVal = sin(angVec.x);

	vector outVec;
	outVec.x =          inpVec.x                    ;
	outVec.y = cosVal * inpVec.y - sinVal * inpVec.z;
	outVec.z = sinVal * inpVec.y + cosVal * inpVec.z;

	return outVec;
}

inline __device__ vector applyRotYMat(vector inpVec, vector angVec)
{
	double cosVal = cos(angVec.y);
	double sinVal = sin(angVec.y);

	vector outVec;
	outVec.x = cosVal * inpVec.x - sinVal * inpVec.z;
	outVec.y =          inpVec.y                    ;
	outVec.z = sinVal * inpVec.x + cosVal * inpVec.z;

	return outVec;
}

inline __device__ vector applyRotZMat(vector inpVec, vector angVec)
{
	double cosVal = cos(angVec.z);
	double sinVal = sin(angVec.z);

	vector outVec;
	outVec.x = cosVal * inpVec.x - sinVal * inpVec.y;
	outVec.y = sinVal * inpVec.x + cosVal * inpVec.y;
	outVec.z =          inpVec.z                    ;

	return outVec;
}

inline __device__ vector applyDRotXMat(vector inpVec, vector angVec)
{
	double cosVal = cos(angVec.x);
	double sinVal = sin(angVec.x);

	vector outVec;
	outVec.x =  0.0                                  ;
	outVec.y = -sinVal * inpVec.y - cosVal * inpVec.z;
	outVec.z =  cosVal * inpVec.y - sinVal * inpVec.z;

	return outVec;
}

inline __device__ vector applyDRotYMat(vector inpVec, vector angVec)
{
	double cosVal = cos(angVec.y);
	double sinVal = sin(angVec.y);

	vector outVec;
	outVec.x = -sinVal * inpVec.x - cosVal * inpVec.z;
	outVec.y =  0.0                                  ;
	outVec.z =  cosVal * inpVec.x - sinVal * inpVec.z;

	return outVec;
}

inline __device__ vector applyDRotZMat(vector inpVec, vector angVec)
{
	double cosVal = cos(angVec.z);
	double sinVal = sin(angVec.z);

	vector outVec;
	outVec.x = -sinVal * inpVec.x - cosVal * inpVec.y;
	outVec.y =  cosVal * inpVec.x - sinVal * inpVec.y;
	outVec.z =  0.0                                  ;

	return outVec;
}

inline __device__ vector applyD2RotXMat(vector inpVec, vector angVec)
{
	double cosVal = cos(angVec.x);
	double sinVal = sin(angVec.x);

	vector outVec;
	outVec.x =  0.0                                  ;
	outVec.y = -cosVal * inpVec.y + sinVal * inpVec.z;
	outVec.z = -sinVal * inpVec.y - cosVal * inpVec.z;

	return outVec;
}

inline __device__ vector applyD2RotYMat(vector inpVec, vector angVec)
{
	double cosVal = cos(angVec.y);
	double sinVal = sin(angVec.y);

	vector outVec;
	outVec.x = -cosVal * inpVec.x + sinVal * inpVec.z;
	outVec.y =  0.0                                  ;
	outVec.z = -sinVal * inpVec.x - cosVal * inpVec.z;

	return outVec;
}

inline __device__ vector applyD2RotZMat(vector inpVec, vector angVec)
{
	double cosVal = cos(angVec.z);
	double sinVal = sin(angVec.z);

	vector outVec;
	outVec.x = -cosVal * inpVec.x + sinVal * inpVec.y;
	outVec.y = -sinVal * inpVec.x - cosVal * inpVec.y;
	outVec.z =  0.0                                  ;

	return outVec;
}

#endif
