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

void render_gate(float3* finaloutputbuffer, float2 offset);