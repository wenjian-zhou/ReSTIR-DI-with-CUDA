#pragma once

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <random>
#include <cmath>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include "cutil_math.h"

#define INV_PI 0.31830988618
#define M_PI 3.14159265359f  // pi
#define scr_width 1024  // screenwidth
#define scr_height 768 // screenheight
#define samps 1 // samples 

struct Reservoir
{
	int id = 0; // Selected light id
	float wSum = 0; // Sum of weights
	int M = 0; // Number of samples

	__device__ void addSample(int lightID, float weight, curandState *randstate)
	{
		M = M + 1;
		wSum = wSum + weight;
		if (curand_uniform(randstate) < weight / wSum)
		{
			id = lightID;
		}
	}
};

void render_gate(float3* finaloutputbuffer, float2 offset, int framenumber, uint hashedframenumber, Reservoir *previousReservoir);

uint WangHash(uint a);