#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifdef DIM2

#define DIMNUM    2
#define VTXNUM    3
#define RGDDOF    3

#elif DIM3

#define DIMNUM    3
#define VTXNUM    4
#define RGDDOF    6

#endif

#define BLKDIM    256
#define BLKROW    16
#define SUMBLKDIM 512

#endif
