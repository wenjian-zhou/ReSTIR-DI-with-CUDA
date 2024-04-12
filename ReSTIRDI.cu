#include "ReSTIRDI.h"

struct Ray 
{
	float3 origin;
	float3 direction;
	__device__ Ray(const float3& m_origin, const float3& m_direction) : origin(m_origin), direction(m_direction) {}
};

enum Refl_t { DIFF, SPEC };

struct Sphere 
{
	float radius;
	float3 position, color;
	Refl_t refl;
	__device__ float Intersect(const Ray& ray) const {
		float3 op = position - ray.origin;
		float t, epsilon = 1e-4;
		float b = dot(op, ray.direction), det = b * b - dot(op, op) + radius * radius;
		if (det < 0) return 0;	else det = sqrtf(det);
		return (t = b - det) > epsilon ? t : ((t = b + det) > epsilon ? t : 0);
	}
};

struct PointLight 
{
	float3 position, emission;
};

__constant__ Sphere spheres[] = {//Scene: radius, position, color
	{1e5, {50, -100000, 0}, {.184,.929,.929}, DIFF},//ground
	{26.5, {22,26.5,42}, {.596,.596,.596}, SPEC},//Ball1
	{13, {75,13,82}, {.91,.29,.102}, DIFF},//Ball2
	{22,{87,22,24}, {.99,.405,.992}, DIFF},//Ball3
	{1e4,{50.0,-40.8,1e4-200}, {.4,.7,.9}, DIFF}//background
};

__constant__ PointLight lights[] = {// Scene: positioin, emission
	{{-50.0, 181.6 + 10.27, 176.6}, {5000.0, 9000.0, 5000.0}},
	{{0.0, 181.6 - 20.27, -126.6}, {1000.0, 5000.0, 7000.0}},
	{{50.0, 181.6 + 305.27, 146.6}, {6000.0, 5000.0, 5000.0}},
	{{80.0, 181.6 - 432.27, -156.6}, {5000.0, 3000.0, 5000.0}},
	{{-20.0, 181.6 + 59.27, 136.6}, {5000.0, 5000.0, 5000.0}},
	{{-10.0, 181.6 - 68.27, 156.6}, {5000.0, 9000.0, 5000.0}},
	{{10.0, 181.6 + 7.27, -106.6}, {5000.0, 5000.0, 9000.0}},
	{{30.0, 181.6 - 8.27, 126.6}, {1000.0, 5000.0, 5000.0}},
	{{60.0, 181.6 + 93.27, -146.6}, {6000.0, 5000.0, 5000.0}},
	{{100.0, 181.6 - 63.27, 166.6}, {5000.0, 3000.0, 5000.0}},
	{{-110.0, 181.6 + 74.27, -161.6}, {1e3, 6e3, 5e3}},
	{{-120.0, 181.6 + 75.27, 152.6}, {4e3, 6e3, 4e3}},
	{{-170.0, 181.6 - 21.27, -236.6}, {1e3, 6e3, 2e3}},
	{{-110.0, 181.6 + 12.27, 256.6}, {6e3, 8e3, 9e3}},
	{{-90.0, 181.6 - 41.27, 137.6}, {3e3, 2e3, 6e3}},
	{{-610.0, 181.6 + 312.27, -115.6}, {8e3, 4e3, 7e3}},
	{{-40.0, 181.6 + 123.27, -235.6}, {2e3, 5e3, 3e3}},
	{{-30.0, 181.6 + 145.27, 236.6}, {1e3, 1e3, 1e3}},
	{{-20.0, 181.6 - 93.27, -132.6}, {6e3, 3e3, 8e3}},
	{{-124.0, 181.6 + 12.27, 152.6}, {2e3, 2e3, 1e3}},
	{{-253.0, 181.6 + 54.27, -523.6}, {4e3, 8e3, 7e3}},
	{{340.0, 181.6 - 14.27, 123.6}, {6e3, 6e3, 1e3}},
	{{230.0, 181.6 + 136.27, -234.6}, {1e3, 3e3, 7e3}},
	{{560.0, 181.6 - 134.27, -45.6}, {7e3, 2e3, 3e3}},
	{{20.0, 181.6 + 34.27, 65.6}, {9e3, 9e3, 1e3}},
	{{80.0, 181.6 - 56.27, -513.6}, {2e3, 7e3, 2e3}},
	{{90.0, 181.6 + 23.27, 34.6}, {4e3, 2e3, 2e3}},
	{{245.0, 181.6 + 77.27, 74.6}, {5e3, 3e3, 6e3}},
	{{54.0, 181.6 + 88.27, -23.6}, {7e3, 8e3, 8e3}},
	{{86.0, 181.6 + 99.27, 45.6}, {3e3, 4e3, 9e3}},
	{{432.0, 181.6 - 14.27, -97.6}, {4e3, 6e3, 2e3}},
	{{744.0, 181.6 + 66.27, -28.6}, {7e3, 2e3, 4e3}},
	{{76.0, 181.6 + 34.27, -534.6}, {8e3, 2e3, 7e3}},
	{{43.0, 181.6 - 76.27, 532.6}, {1e3, 8e3, 3e3}},
	{{258.0, 181.6 + 85.27, -93.6}, {4e3, 7e3, 7e3}},
	{{224.0, 181.6 + 24.27, -76.6}, {8e3, 4e3, 3e3}},
	{{13.0, 181.6 + 45.27, 36.6}, {3e3, 7e3, 8e3}},
	{{64.0, 181.6 - 46.27, -34.6}, {6e3, 4e3, 5e3}},
	{{99.0, 181.6 - 32.27, -45.6}, {8e3, 6e3, 1e3}}
};

inline __host__ __device__ float Clamp(float x) 
{
	return x < 0 ? 0 : x > 1 ? 1 : x;
}

inline __host__ __device__ int ToInt(float x) 
{
	return int(pow(Clamp(x), 1 / 2.2) * 255 + .5);
}

inline __device__ bool Intersect(const Ray& ray, float& t, int& id) 
{
	float n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
	for (int i = int(n); i--;) if ((d = spheres[i].Intersect(ray)) && d < t)
	{
		t = d, id = i;
	}
	return t < inf;
}

__device__ void RISAndTemporalReuse(const int &frameNumber, const Ray& r, const int& M, curandState *randstate, int2 index, int pixelIndex, 
							Reservoir *previousReservoir, Reservoir *currentReservoir,
							bool temporalReuse, bool spatialReuse)
{
	float t; // distance to intersection
	int id = 0; // id of intersected object
	Ray ray = r;

	if (!Intersect(ray, t, id)) return;
	Sphere* obj = &spheres[id]; // the hit object
	float3 hitPoint = ray.origin + ray.direction * t;
	float3 normal = normalize(hitPoint - obj->position);
	float3 normalLocal = dot(normal, ray.direction) < 0 ? normal : normal * -1;

	// Perform specular reflection but not as iteration
	if (obj->refl == SPEC)
	{
		float3 reflectionDirection = ray.direction - normal * 2 * dot(normal, ray.direction);
		Ray reflectionRay = Ray(hitPoint + normal * 2e-2, reflectionDirection);
		if (!Intersect(reflectionRay, t, id)) return ;
		obj = &spheres[id]; // the hit object
		hitPoint = reflectionRay.origin + reflectionRay.direction * t;
		normal = normalize(hitPoint - obj->position);
		normalLocal = dot(normal, reflectionRay.direction) < 0 ? normal : normal * -1;
	}

	int lightsCount = sizeof(lights) / sizeof(PointLight);
	int RISSamples = lightsCount > M ? M : lightsCount;

	Reservoir reservoir;
	Reservoir visibility_reservoir;
	Reservoir temporal_reservoir;

	reservoir.normal = normalLocal;
	reservoir.depth = t;
	visibility_reservoir.normal = normalLocal;
	visibility_reservoir.depth = t;
	temporal_reservoir.normal = normalLocal;
	temporal_reservoir.depth = t;

	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////// Candidate Generation ////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////

	for (int i = 0; i < RISSamples; i++)
	{
		// Pick a random light from the scene to sample
		int randomLight = int(curand_uniform(randstate) * lightsCount);
		int lightToSample = randomLight > lightsCount - 1 ? lightsCount - 1 : randomLight;

		// Sample the light
		float distanceToLight = length(lights[lightToSample].position - hitPoint);
		float3 lightEmission = lights[lightToSample].emission;
		float3 lightDirection = normalize(lights[lightToSample].position - hitPoint);

		// Compute the Lambertian cosine
		float cosTheta = dot(normalLocal, lightDirection);
		if (cosTheta < 0) cosTheta = 0.;

		// Calculate the light attenuation
		float lightAttenuation = 1 / (distanceToLight * distanceToLight);

		// Compute the BRDF
		float3 BRDF = obj->color * INV_PI * cosTheta;

		// Compute the light intensity
		float3 lightIntensity = lightEmission * lightAttenuation;

		// Compute radiance
		float3 radiance = BRDF * lightIntensity;

		// Compute pHat
		float pHat = length(radiance);

		// MIS weight
		float MISWeight = 1. / RISSamples;

		// Contribution weight of light
		float contributionWeight = lightsCount; // The inverse PDF of sampling the light

		// The weight of the sample
		float weight = pHat * MISWeight * contributionWeight;

		// Add the sample to the reservoir
		reservoir.addSample(lightToSample, weight, randstate);
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////  Visibility Pass ///////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////

	// Check if the reservoir sample is visible
	int tempLight = reservoir.id;
	float v_distanceToLight = length(lights[tempLight].position - hitPoint);
	float3 v_lightEmission = lights[tempLight].emission;
	float3 v_lightDirection = normalize(lights[tempLight].position - hitPoint);
	float v_cosTheta = dot(normalLocal, v_lightDirection);
	if (v_cosTheta < 0) v_cosTheta = 0.;
	float v_visibility = 0;
	Ray v_shadowRay(hitPoint + normalLocal * 2e-2, v_lightDirection);
	if (!(Intersect(v_shadowRay, t, id) && t < length(lights[tempLight].position - hitPoint) - 2e-2))
	{
		v_visibility = 1;
	}
	float v_lightAttenuation = 1 / (v_distanceToLight * v_distanceToLight);
	float3 v_BRDF = obj->color * INV_PI * v_cosTheta;
	float3 v_lightIntensity = v_lightEmission * v_lightAttenuation;
	float3 v_radiance = v_BRDF * v_lightIntensity * v_visibility;
	float v_pHat = length(v_radiance);
	float v_weight = v_pHat > 0.0 ? (1. / v_pHat) * reservoir.wSum : 0.0;
	float v_ContributionWeight = v_pHat * v_weight * v_visibility;

	visibility_reservoir.addSample(tempLight, v_ContributionWeight, randstate);
	reservoir.id = visibility_reservoir.id;
	reservoir.wSum = visibility_reservoir.wSum;

	// Update the current reservoir
	currentReservoir[pixelIndex] = reservoir;

	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////// Temporal Reuse Pass /////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////

	if (temporalReuse && frameNumber > 1)
	{
		int currentSample = reservoir.id;

		int previousSample = previousReservoir[pixelIndex].id;

		// calculate current pHat
		float distanceToLight = length(lights[currentSample].position - hitPoint);
		float3 lightEmission = lights[currentSample].emission;
		float3 lightDirection = normalize(lights[currentSample].position - hitPoint);
		float cosTheta = dot(normalLocal, lightDirection);
		if (cosTheta < 0) cosTheta = 0.;
		float lightAttenuation = 1 / (distanceToLight * distanceToLight);
		float3 BRDF = obj->color * INV_PI * cosTheta;
		float3 lightIntensity = lightEmission * lightAttenuation;
		float3 radiance = BRDF * lightIntensity;
		float currPHat = length(radiance);

		// calculate previous pHat
		distanceToLight = length(lights[previousSample].position - hitPoint);
		lightEmission = lights[previousSample].emission;
		lightDirection = normalize(lights[previousSample].position - hitPoint);
		cosTheta = dot(normalLocal, lightDirection);
		if (cosTheta < 0) cosTheta = 0.;
		lightAttenuation = 1 / (distanceToLight * distanceToLight);
		BRDF = obj->color * INV_PI * cosTheta;
		lightIntensity = lightEmission * lightAttenuation;
		radiance = BRDF * lightIntensity;
		float prevPHat = length(radiance);

		// calculate MIS weights for both samples
		float currentMISWeight = currPHat / (20.0 * prevPHat + currPHat);
		float previousMISWeight = 20.0 * prevPHat / (20.0 * prevPHat + currPHat);

		// calculate the weight of the samples
		float currentSampleWeight = currPHat > 0.0 ? (1. / currPHat) * reservoir.wSum : 0.0;
		float previousSampleWeight = prevPHat > 0.0 ? (1. / prevPHat) * previousReservoir[pixelIndex].wSum : 0.0;

		// calculate the contribution weight of the samples
		float currentContributionWeight = currPHat * currentMISWeight * currentSampleWeight;
		float previousContributionWeight = prevPHat * previousMISWeight * previousSampleWeight;

		temporal_reservoir.addSample(currentSample, currentContributionWeight, randstate);
		temporal_reservoir.addSample(previousSample, previousContributionWeight, randstate);

		// Update the reservoir
		if (temporal_reservoir.wSum > 0.0)
		{
			reservoir.id = temporal_reservoir.id;
			reservoir.wSum = temporal_reservoir.wSum;
		}	
	}

	// Update the current and previous reservoir
	currentReservoir[pixelIndex] = reservoir;
	previousReservoir[pixelIndex] = reservoir;
}

__device__ float3 SpatialReuseAndFinalColor(const int &frameNumber, const Ray& r, const int& M, curandState *randstate, int2 index, int pixelIndex, 
							Reservoir *previousReservoir, Reservoir *currentReservoir,
							bool temporalReuse, bool spatialReuse)
{
	float t; // distance to intersection
	int id = 0; // id of intersected object
	Ray ray = r;

	if (!Intersect(ray, t, id)) return make_float3(0.0f, 0.0f, 0.0f);
	Sphere* obj = &spheres[id]; // the hit object
	float3 hitPoint = ray.origin + ray.direction * t;
	float3 normal = normalize(hitPoint - obj->position);
	float3 normalLocal = dot(normal, ray.direction) < 0 ? normal : normal * -1;
	bool specularReflection = false;

	// Perform specular reflection but not as iteration
	if (obj->refl == SPEC)
	{
		float3 reflectionDirection = ray.direction - normal * 2 * dot(normal, ray.direction);
		Ray reflectionRay = Ray(hitPoint + normal * 2e-2, reflectionDirection);
		if (!Intersect(reflectionRay, t, id)) return make_float3(0.0f, 0.0f, 0.0f);
		obj = &spheres[id]; // the hit object
		hitPoint = reflectionRay.origin + reflectionRay.direction * t;
		normal = normalize(hitPoint - obj->position);
		normalLocal = dot(normal, reflectionRay.direction) < 0 ? normal : normal * -1;
		specularReflection = true;
	}

	Reservoir reservoir;
	Reservoir spatial_reservoir;

	reservoir.normal = normalLocal;
	reservoir.depth = t;
	spatial_reservoir.normal = normalLocal;
	spatial_reservoir.depth = t;
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////// Spatial Reuse Pass /////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////

	reservoir = currentReservoir[pixelIndex];

	if (spatialReuse && frameNumber > 1)
	{
		const uint k = 5; // total number of the spatial neighborhood
		uint finalSpatialNeighborCount = 5; // number of the spatial neighborhood that are similar
		uint range; // range of the spatial neighborhood
		if (!specularReflection)
		{
			range = 15;
		}
		else
		{
			range = 1;
		}
		int2 offset = make_int2(0, 0);
		int spatialSampleIndex[k];

		for (int i = 0; i < k; i ++)
		{
			offset.x = i > 0 ? int(curand_uniform(randstate) * 2.0 * range) - range : 0;
			offset.y = i > 0 ? int(curand_uniform(randstate) * 2.0 * range) - range : 0;

			int2 neighborIndex = max(make_int2(0, 0), min(index + offset, make_int2(scr_width - 1, scr_height - 1)));
			spatialSampleIndex[i] = (scr_height - neighborIndex.y - 1) * scr_width + neighborIndex.x;

			if (dot(reservoir.normal, currentReservoir[spatialSampleIndex[i]].normal) < 0.9 
				 && abs(reservoir.depth - currentReservoir[spatialSampleIndex[i]].depth) >= 0.1)
			{
				finalSpatialNeighborCount = finalSpatialNeighborCount - 1;
				spatialSampleIndex[i] = -1;
			}
		}

		// calculate MIS weight
		float spatialMISWeight = 1.f / (float)finalSpatialNeighborCount;

		for (int i = 0; i < k; i ++)
		{
			// skip dissimilar samples
			if (spatialSampleIndex[i] == -1)
			{
				continue;
			}

			int spatialSample = currentReservoir[spatialSampleIndex[i]].id;

			// calculate spatial pHat
			float distanceToLight = length(lights[spatialSample].position - hitPoint);
			float3 lightEmission = lights[spatialSample].emission;
			float3 lightDirection = normalize(lights[spatialSample].position - hitPoint);
			float cosTheta = dot(normalLocal, lightDirection);
			if (cosTheta < 0) cosTheta = 0.;
			// Check if the light is visible
			float visibility = 0;
			Ray shadowRay(hitPoint + normalLocal * 2e-2, lightDirection);
			if (!(Intersect(shadowRay, t, id) && t < distanceToLight - 2e-2))
			{
				visibility = 1;
			}
			float lightAttenuation = 1 / (distanceToLight * distanceToLight);
			float3 BRDF = obj->color * INV_PI * cosTheta;
			float3 lightIntensity = lightEmission * lightAttenuation;
			float3 radiance = BRDF * lightIntensity * visibility;
			float spatialPHat = length(radiance);

			// calculate the weight of the sample
			float spatialSampleWeight = spatialPHat > 0.0 ? (1. / spatialPHat) * currentReservoir[spatialSampleIndex[i]].wSum : 0.0;

			// calculate the contribution weight of the sample
			float spatialContributionWeight = spatialMISWeight * spatialPHat * spatialSampleWeight;

			// add sample to spatial reservoir
			spatial_reservoir.addSample(spatialSample, spatialContributionWeight, randstate);
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////// Final Color /////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////

	// The chosen light to sample and the contribution weight
	int finalLight = spatialReuse ? spatial_reservoir.id : reservoir.id;

	// Sample the light
	float distanceToLight = length(lights[finalLight].position - hitPoint);
	float3 lightEmission = lights[finalLight].emission;
	float3 lightDirection = normalize(lights[finalLight].position - hitPoint);
	
	// Compute the Lambertian cosine
	float cosTheta = dot(normalLocal, lightDirection);
	if (cosTheta < 0) cosTheta = 0.;

	// Check if the light is visible
	float visibility = 0;
	Ray shadowRay(hitPoint + normalLocal * 2e-2, lightDirection);
	if (!(Intersect(shadowRay, t, id) && t < distanceToLight - 2e-2))
	{
		visibility = 1;
	}

	// Calculate the light attenuation
	float lightAttenuation = 1 / (distanceToLight * distanceToLight);

	// Compute the BRDF
	float3 BRDF = obj->color * INV_PI * cosTheta;

	// Compute the light intensity
	float3 lightIntensity = lightEmission * lightAttenuation;

	// Compute radiance
	float3 radiance = BRDF * lightIntensity * visibility;

	// Compute pHat
	float pHat = length(radiance);

	// pHat multiplied by the visibility
	float weight = pHat > 0.0 ? (1. / pHat) * (spatialReuse ? spatial_reservoir.wSum : reservoir.wSum) : 0.0;

	// Compute the direct illumination of Lambertian BRDF
	float3 color = radiance * weight;
	
	return color;
}

__device__ float3 DirectIllumination(const Ray& r, curandState *randstate)
{
	float t; // distance to intersection
	int id = 0; // id of intersected object
	Ray ray = r;
	
	if (!Intersect(ray, t, id)) return make_float3(0.0f, 0.0f, 0.0f);
	Sphere* obj = &spheres[id]; // the hit object
	float3 hitPoint = ray.origin + ray.direction * t;
	float3 normal = normalize(hitPoint - obj->position);
	float3 normalLocal = dot(normal, ray.direction) < 0 ? normal : normal * -1;

	// Perform specular reflection but not as iteration
	if (obj->refl == SPEC)
	{
		float3 reflectionDirection = ray.direction - normal * 2 * dot(normal, ray.direction);
		Ray reflectionRay = Ray(hitPoint + normal * 2e-2, reflectionDirection);
		if (!Intersect(reflectionRay, t, id)) return make_float3(0.0f, 0.0f, 0.0f);
		obj = &spheres[id]; // the hit object
		hitPoint = reflectionRay.origin + reflectionRay.direction * t;
		normal = normalize(hitPoint - obj->position);
		normalLocal = dot(normal, reflectionRay.direction) < 0 ? normal : normal * -1;
	}

	// Pick a random light from the scene to sample
	int lightsCount = sizeof(lights) / sizeof(PointLight);
	int randomLight = int(curand_uniform(randstate) * lightsCount);
	int lightToSample = randomLight > lightsCount - 1 ? lightsCount - 1 : randomLight;

	// Sample the light
	float distanceToLight = length(lights[lightToSample].position - hitPoint);
	float3 lightEmission = lights[lightToSample].emission;
	float3 lightDirection = normalize(lights[lightToSample].position - hitPoint);

	// Compute the Lambertian cosine
	float cosTheta = dot(normalLocal, lightDirection);
	if (cosTheta < 0) return make_float3(0.0f, 0.0f, 0.0f);

	// Check if the light is visible
	float visibility = 0;
	Ray shadowRay(hitPoint + normalLocal * 2e-2, lightDirection);
	if (!(Intersect(shadowRay, t, id) && t < distanceToLight - 2e-2))
	{
		visibility = 1;
	}

	// Calculate the inverse PDF of sampling the light
	float invPdf = (float)lightsCount;

	// Calculate the light attenuation
	float lightAttenuation = 1 / (distanceToLight * distanceToLight);

	// Compute the direct illumination of Lambertian BRDF
	float3 shadingColor = make_float3(0.0f, 0.0f, 0.0f);
	shadingColor += obj->color * lightEmission * INV_PI * cosTheta * lightAttenuation * visibility * invPdf;

	return shadingColor;
}

// union struct required for mapping pixel colours to OpenGL buffer
union Colour  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

// hash function to calculate new seed for each frame
// see http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
uint WangHash(uint a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

__global__ void RISAndTemporalReuseKernel(float3 *finalOutputBuffer, int frameNumber, uint hashedFrameNumber, 
							  Reservoir *previousReservoir, Reservoir *currentReservoir, 
							  bool useReSTIR, bool temporalReuse, bool spatialReuse) {
	// assign a CUDA thread to every pixel (x,y) 
    // blockIdx, blockDim and threadIdx are CUDA specific keywords
    // replaces nested outer loops in CPU code looping over image rows and image columns 
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int i = (scr_height - y - 1)*scr_width + x; // index of current pixel (calculated using thread index) 

	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// create random number generator, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedFrameNumber + threadId, 0, 0, &randState);

	Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // first hardcoded camera ray(origin, direction)
	float3 cx = make_float3(scr_width * .5135 / scr_height, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, cam.direction)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)
        
	// compute primary ray direction
	float3 d = cam.direction + cx*((.25 + x) / scr_width - .5) + cy*((.25 + y) / scr_height - .5);
		
	// create primary ray, add incoming radiance to pixelcolor

	RISAndTemporalReuse(frameNumber, Ray(cam.origin + d * 40, normalize(d)), 32, &randState, make_int2(x, y), i, 
						  previousReservoir, currentReservoir, 
						  temporalReuse, spatialReuse); 
}

__global__ void SpatialReuseAndFinalColorKernel(float3 *finalOutputBuffer, int frameNumber, uint hashedFrameNumber, 
							  Reservoir *previousReservoir, Reservoir *currentReservoir, 
							  bool useReSTIR, bool temporalReuse, bool spatialReuse) {
	// assign a CUDA thread to every pixel (x,y) 
    // blockIdx, blockDim and threadIdx are CUDA specific keywords
    // replaces nested outer loops in CPU code looping over image rows and image columns 
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int i = (scr_height - y - 1)*scr_width + x; // index of current pixel (calculated using thread index) 

	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// create random number generator, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedFrameNumber + threadId, 0, 0, &randState);

	Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // first hardcoded camera ray(origin, direction)
	float3 cx = make_float3(scr_width * .5135 / scr_height, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, cam.direction)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)
    float3 r; // r is final pixel color     

	r = make_float3(0.0f); // reset r to zero for every pixel 
        
	// compute primary ray direction
	float3 d = cam.direction + cx*((.25 + x) / scr_width - .5) + cy*((.25 + y) / scr_height - .5);
		
	// create primary ray, add incoming radiance to pixelcolor
	r = r + SpatialReuseAndFinalColor(frameNumber, Ray(cam.origin + d * 40, normalize(d)), 32, &randState, make_int2(x, y), i, 
						  previousReservoir, currentReservoir, 
						  temporalReuse, spatialReuse); 

	Colour fcolour;
	float3 colour = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
	
	// convert from 96-bit to 24-bit colour + perform gamma correction
  	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255),
    (unsigned char)(powf(colour.y, 1 / 2.2f) * 255),
    (unsigned char)(powf(colour.z, 1 / 2.2f) * 255),1);

	finalOutputBuffer[i] = make_float3(x, y, fcolour.c);
}

__global__ void DIKernel(float3 *finalOutputBuffer, int frameNumber, uint hashedFrameNumber, 
							  Reservoir *previousReservoir, Reservoir *currentReservoir, 
							  bool useReSTIR, bool temporalReuse, bool spatialReuse) {
	// assign a CUDA thread to every pixel (x,y) 
    // blockIdx, blockDim and threadIdx are CUDA specific keywords
    // replaces nested outer loops in CPU code looping over image rows and image columns 
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int i = (scr_height - y - 1)*scr_width + x; // index of current pixel (calculated using thread index) 

	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// create random number generator, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedFrameNumber + threadId, 0, 0, &randState);

	Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // first hardcoded camera ray(origin, direction)
	float3 cx = make_float3(scr_width * .5135 / scr_height, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, cam.direction)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)
    float3 r; // r is final pixel color     

	r = make_float3(0.0f); // reset r to zero for every pixel 
        
	// compute primary ray direction
	float3 d = cam.direction + cx*((.25 + x) / scr_width - .5) + cy*((.25 + y) / scr_height - .5);
		
	// create primary ray, add incoming radiance to pixelcolor
	r = r + DirectIllumination(Ray(cam.origin + d * 40, normalize(d)), &randState);

	Colour fcolour;
	float3 colour = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
	
	// convert from 96-bit to 24-bit colour + perform gamma correction
  	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255),
    (unsigned char)(powf(colour.y, 1 / 2.2f) * 255),
    (unsigned char)(powf(colour.z, 1 / 2.2f) * 255),1);

	finalOutputBuffer[i] = make_float3(x, y, fcolour.c);
}

void RenderGate(float3* finalOutputBuffer, int frameNumber, uint hashedFrameNumber, 
				 Reservoir *previousReservoir, Reservoir *currentReservoir, 
				 bool useReSTIR, bool temporalReuse, bool spatialReuse) {
    // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
    dim3 block(16, 16, 1);   
    dim3 grid(scr_width / block.x, scr_height / block.y, 1);
    
    // schedule threads on device and launch CUDA kernel from host
	if (useReSTIR)
	{
		RISAndTemporalReuseKernel <<< grid, block >>>(finalOutputBuffer, frameNumber, hashedFrameNumber, 
									  previousReservoir, currentReservoir, 
									  useReSTIR, temporalReuse, spatialReuse);
		SpatialReuseAndFinalColorKernel <<< grid, block >>>(finalOutputBuffer, frameNumber, hashedFrameNumber, 
									  previousReservoir, currentReservoir, 
									  useReSTIR, temporalReuse, spatialReuse);
	}
	else
	{
    	DIKernel <<< grid, block >>>(finalOutputBuffer, frameNumber, hashedFrameNumber, 
									  previousReservoir, currentReservoir, 
									  useReSTIR, temporalReuse, spatialReuse);  
	}
	// Wait for GPU to finish before accessing on host
  	cudaDeviceSynchronize();
}

__device__ static float getrandom(unsigned int *seed0, unsigned int *seed1) {
    *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
    *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

    unsigned int ires = ((*seed0) << 16) + (*seed1);

    // Convert to float
    union {
    float f;
    unsigned int ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

    return (res.f - 2.f) / 2.f;
}

__device__ float3 Reference_DI(const Ray& r, unsigned int *s1, unsigned int *s2)
{
	float t; // distance to intersection
	int id = 0; // id of intersected object
	Ray ray = r;
	
	if (!Intersect(ray, t, id)) return make_float3(0.0f, 0.0f, 0.0f);
	Sphere* obj = &spheres[id]; // the hit object
	float3 hitPoint = ray.origin + ray.direction * t;
	float3 normal = normalize(hitPoint - obj->position);
	float3 normalLocal = dot(normal, ray.direction) < 0 ? normal : normal * -1;

	// Perform specular reflection but not as iteration
	if (obj->refl == SPEC)
	{
		float3 reflectionDirection = ray.direction - normal * 2 * dot(normal, ray.direction);
		Ray reflectionRay = Ray(hitPoint + normal * 2e-2, reflectionDirection);
		if (!Intersect(reflectionRay, t, id)) return make_float3(0.0f, 0.0f, 0.0f);
		obj = &spheres[id]; // the hit object
		hitPoint = reflectionRay.origin + reflectionRay.direction * t;
		normal = normalize(hitPoint - obj->position);
		normalLocal = dot(normal, reflectionRay.direction) < 0 ? normal : normal * -1;
	}

	// Pick a random light from the scene to sample
	int lightsCount = sizeof(lights) / sizeof(PointLight);
	int randomLight = int(getrandom(s1, s2) * lightsCount);
	int lightToSample = randomLight > lightsCount - 1 ? lightsCount - 1 : randomLight;

	// Sample the light
	float distanceToLight = length(lights[lightToSample].position - hitPoint);
	float3 lightEmission = lights[lightToSample].emission;
	float3 lightDirection = normalize(lights[lightToSample].position - hitPoint);

	// Compute the Lambertian cosine
	float cosTheta = dot(normalLocal, lightDirection);
	if (cosTheta < 0) return make_float3(0.0f, 0.0f, 0.0f);

	// Check if the light is visible
	float visibility = 0;
	Ray shadowRay(hitPoint + normalLocal * 2e-2, lightDirection);
	if (!(Intersect(shadowRay, t, id) && t < distanceToLight - 2e-2))
	{
		visibility = 1;
	}

	// Calculate the inverse PDF of sampling the light
	float invPdf = (float)lightsCount;

	// Calculate the light attenuation
	float lightAttenuation = 1 / (distanceToLight * distanceToLight);

	// Compute the direct illumination of Lambertian BRDF
	float3 shadingColor = make_float3(0.0f, 0.0f, 0.0f);
	shadingColor += obj->color * lightEmission * INV_PI * cosTheta * lightAttenuation * visibility * invPdf;

	return shadingColor;
}

__global__ void reference_kernel(float3 *output) {
	// assign a CUDA thread to every pixel (x,y) 
    // blockIdx, blockDim and threadIdx are CUDA specific keywords
    // replaces nested outer loops in CPU code looping over image rows and image columns 
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int i = (scr_height - y - 1)*scr_width + x; // index of current pixel (calculated using thread index) 

	unsigned int s1 = x;
	unsigned int s2 = y;

	Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // first hardcoded camera ray(origin, direction)
	float3 cx = make_float3(scr_width * .5135 / scr_height, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, cam.direction)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)
    float3 r; // r is final pixel color     

	r = make_float3(0.0f); // reset r to zero for every pixel 
        
	// compute primary ray direction
	float3 d = cam.direction + cx*((.25 + x) / scr_width - .5) + cy*((.25 + y) / scr_height - .5);
		
	// create primary ray, add incoming radiance to pixelcolor
	for (int s = 0; s < 32768; s ++)
	{
		r = r + Reference_DI(Ray(cam.origin + d * 40, normalize(d)), &s1, &s2) / 32768.0;
	}

	float3 colour = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));

	output[i] = colour;
}

void produceReference()
{
	float3* output_h = new float3[scr_width * scr_height]; // allocate memory for the image on the host
	float3* output_d; // allocate memory for the image on the device

	// allocate memory on the device
	cudaMalloc(&output_d, scr_width * scr_height * sizeof(float3));

	dim3 block(16, 16, 1); // block dimensions
	dim3 grid(scr_width / block.x, scr_height / block.y, 1); // grid dimensions

	printf("Rendering reference image...\n");

	reference_kernel <<< grid, block >>>(output_d); // schedule threads on device and launch CUDA kernel from host

	cudaMemcpy(output_h, output_d, scr_width * scr_height * sizeof(float3), cudaMemcpyDeviceToHost); // copy the result back to the host

	cudaFree(output_d); // free memory on the device

	printf("Reference image rendered.\n");

	// write the image to a file
	FILE *f = fopen("reference.ppm", "w");          
    fprintf(f, "P3\n%d %d\n%d\n", scr_width, scr_height, 255);
    for (int i = 0; i < scr_width*scr_height; i++)  // loop over pixels, write RGB values
    fprintf(f, "%d %d %d ", ToInt(output_h[i].x),
                            ToInt(output_h[i].y),
                            ToInt(output_h[i].z));

    printf("Saved image to 'reference.ppm'\n");

    delete[] output_h;
}