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
	float wSum = 0.0; // Sum of weights
	float3 normal = make_float3(0, 0, 0); // Normal
	float depth = 0.0; // Depth

	__device__ void addSample(int lightID, float weight, curandState *randstate)
	{
		wSum = wSum + weight;
		if (curand_uniform(randstate) < weight / wSum)
		{
			id = lightID;
		}
	}
};

void RenderGate(float3* finalOutputBuffer, int frameNumber, uint hashedFrameNumber, 
				 Reservoir *previousReservoir, Reservoir *currentReservoir, 
				 bool useReSTIR, bool temporalReuse, bool spatialReuse);

uint WangHash(uint a);

void produceReference();