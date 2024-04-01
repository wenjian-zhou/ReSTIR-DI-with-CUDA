#include "ReSTIRDI.h"

// random number generator from https://github.com/gz/rust-raytracer

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
	__device__ double Intersect(const Ray& ray) const {
		float3 op = position - ray.origin;
		double t, epsilon = 1e-4;
		double b = dot(op, ray.direction), det = b * b - dot(op, op) + radius * radius;
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
	{4e4, {50, -4e4-30, -3000}, {.2,.2,.2}, DIFF},//mountains
	{26.5, {22,26.5,42}, {.596,.596,.596}, SPEC},//Ball1
	{13, {75,13,82}, {.96,.96,.96}, DIFF},//Ball2
	{22,{87,22,24}, {.75,.55,.25}, DIFF},//Ball3
	{1e4,{50.0,40.8,-1e4-200}, {.7,.7,.7}, DIFF}//Sky
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
	{{-110.0, 181.6 + 77.27, -156.6}, {1e3, 6e3, 1e3}}
};

inline __host__ __device__ double Clamp(double x) 
{
	return x < 0 ? 0 : x > 1 ? 1 : x;
}

inline __host__ __device__ int ToInt(double x) 
{
	return int(pow(Clamp(x), 1 / 2.2) * 255 + .5);
}

inline __device__ bool Intersect(const Ray& ray, double& t, int& id) 
{
	double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
	for (int i = int(n); i--;) if ((d = spheres[i].Intersect(ray)) && d < t)
	{
		t = d, id = i;
	}
	return t < inf;
}

struct Reservoir
{
	int id = 0; // Selected light id
	float wSum = 0; // Sum of weights

	__device__ void addSample(int lightID, float weight, unsigned int *s1, unsigned int *s2)
	{
		wSum = wSum + weight;
		if (getrandom(s1, s2) < weight / wSum)
		{
			id = lightID;
		}
	}
};

__device__ float3 RIS_DI(const Ray& r, const int& M, unsigned int *s1, unsigned int *s2)
{
	double t; // distance to intersection
	int id = 0; // id of intersected object
	Ray ray = r;

	if (!Intersect(ray, t, id)) return make_float3(0.0f, 0.0f, 0.0f);
	Sphere* obj = &spheres[id]; // the hit object
	float3 hitPoint = ray.origin + ray.direction * t;
	float3 normal = normalize(hitPoint - obj->position);
	float3 normal_local = dot(normal, ray.direction) < 0 ? normal : normal * -1;

	// Perform specular reflection but not as iteration
	if (obj->refl == SPEC)
	{
		float3 reflectionDirection = ray.direction - normal * 2 * dot(normal, ray.direction);
		Ray reflectionRay = Ray(hitPoint + normal * 2e-2, reflectionDirection);
		if (!Intersect(reflectionRay, t, id)) return make_float3(0.0f, 0.0f, 0.0f);
		obj = &spheres[id]; // the hit object
		hitPoint = reflectionRay.origin + reflectionRay.direction * t;
		normal = normalize(hitPoint - obj->position);
		normal_local = dot(normal, reflectionRay.direction) < 0 ? normal : normal * -1;
	}

	int lightsCount = sizeof(lights) / sizeof(PointLight);
	int RISSamples = lightsCount > M ? M : lightsCount;

	Reservoir reservoir;

	for (int i = 0; i < RISSamples; i++)
	{
		// Pick a random light from the scene to sample
		int randomLight = int(getrandom(s1, s2) * lightsCount);
		int lightToSample = randomLight > lightsCount - 1 ? lightsCount - 1 : randomLight;

		// Sample the light
		double distanceToLight = length(lights[lightToSample].position - hitPoint);
		float3 lightEmission = lights[lightToSample].emission;
		float3 lightDirection = normalize(lights[lightToSample].position - hitPoint);

		// Compute the Lambertian cosine
		double cosTheta = dot(normal_local, lightDirection);
		if (cosTheta < 0) cosTheta = 0.;

		// Calculate the light attenuation
		double lightAttenuation = 1 / (distanceToLight * distanceToLight);

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
		reservoir.addSample(lightToSample, weight, s1, s2);
	}

	// The chosen light to sample and the contribution weight
	int finalLight = reservoir.id;
	//printf("Final light: %d\n", finalLight);

	// Sample the light
	double distanceToLight = length(lights[finalLight].position - hitPoint);
	float3 lightEmission = lights[finalLight].emission;
	float3 lightDirection = normalize(lights[finalLight].position - hitPoint);
	
	// Compute the Lambertian cosine
	double cosTheta = dot(normal_local, lightDirection);
	if (cosTheta < 0) cosTheta = 0.;

	// Check if the light is visible
	double visibility = 0;
	Ray shadowRay(hitPoint + normal_local * 2e-2, lightDirection);
	if (!(Intersect(shadowRay, t, id) && t < distanceToLight - 2e-2))
	{
		visibility = 1;
	}

	// Calculate the light attenuation
	double lightAttenuation = 1 / (distanceToLight * distanceToLight);

	// Compute the BRDF
	float3 BRDF = obj->color * INV_PI * cosTheta;

	// Compute the light intensity
	float3 lightIntensity = lightEmission * lightAttenuation;

	// Compute radiance
	float3 radiance = BRDF * lightIntensity * visibility;

	// Compute pHat
	float pHat = length(radiance);

	// pHat multiplied by the visibility
	float weight = pHat > 0.0 ? (1. / pHat) * reservoir.wSum : 0.0;

	// Compute the direct illumination of Lambertian BRDF
	float3 color = radiance * weight;

	return color;
}

__device__ float3 DirectIllumination(const Ray& r, unsigned int *s1, unsigned int *s2)
{
	double t; // distance to intersection
	int id = 0; // id of intersected object
	Ray ray = r;
	
	if (!Intersect(ray, t, id)) return make_float3(0.0f, 0.0f, 0.0f);
	Sphere* obj = &spheres[id]; // the hit object
	float3 hitPoint = ray.origin + ray.direction * t;
	float3 normal = normalize(hitPoint - obj->position);
	float3 normal_local = dot(normal, ray.direction) < 0 ? normal : normal * -1;

	// Perform specular reflection but not as iteration
	if (obj->refl == SPEC)
	{
		float3 reflectionDirection = ray.direction - normal * 2 * dot(normal, ray.direction);
		Ray reflectionRay = Ray(hitPoint + normal * 2e-2, reflectionDirection);
		if (!Intersect(reflectionRay, t, id)) return make_float3(0.0f, 0.0f, 0.0f);
		obj = &spheres[id]; // the hit object
		hitPoint = reflectionRay.origin + reflectionRay.direction * t;
		normal = normalize(hitPoint - obj->position);
		normal_local = dot(normal, reflectionRay.direction) < 0 ? normal : normal * -1;
	}

	// Pick a random light from the scene to sample
	int lightsCount = sizeof(lights) / sizeof(PointLight);
	int randomLight = int(getrandom(s1, s2) * lightsCount);
	int lightToSample = randomLight > lightsCount - 1 ? lightsCount - 1 : randomLight;

	// Sample the light
	double distanceToLight = length(lights[lightToSample].position - hitPoint);
	float3 lightEmission = lights[lightToSample].emission;
	float3 lightDirection = normalize(lights[lightToSample].position - hitPoint);

	// Compute the Lambertian cosine
	double cosTheta = dot(normal_local, lightDirection);
	if (cosTheta < 0) return make_float3(0.0f, 0.0f, 0.0f);

	// Check if the light is visible
	double visibility = 0;
	Ray shadowRay(hitPoint + normal_local * 2e-2, lightDirection);
	if (!(Intersect(shadowRay, t, id) && t < distanceToLight - 2e-2))
	{
		visibility = 1;
	}

	// Calculate the inverse PDF of sampling the light
	double invPdf = (double)lightsCount;

	// Calculate the light attenuation
	double lightAttenuation = 1 / (distanceToLight * distanceToLight);

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

__global__ void render_kernel(float3 *finaloutputbuffer, float2 offset) {
	// assign a CUDA thread to every pixel (x,y) 
    // blockIdx, blockDim and threadIdx are CUDA specific keywords
    // replaces nested outer loops in CPU code looping over image rows and image columns 
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int i = (scr_height - y - 1)*scr_width + x; // index of current pixel (calculated using thread index) 

    unsigned int s1 = x;  // seeds for random number generator
    unsigned int s2 = y;

	Ray cam(make_float3(50 + offset.x * 10, 52 - offset.y * 10, 295.6), normalize(make_float3(0, -0.042612, -1))); // first hardcoded camera ray(origin, direction)
	float3 cx = make_float3(scr_width * .5135 / scr_height, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, cam.direction)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)
    float3 r; // r is final pixel color     

	r = make_float3(0.0f); // reset r to zero for every pixel 

    for (int s = 0; s < samps; s++){  // samples per pixel
        
		// compute primary ray direction
		float3 d = cam.direction + cx*((.25 + x) / scr_width - .5) + cy*((.25 + y) / scr_height - .5);
		
		// create primary ray, add incoming radiance to pixelcolor
		if (x <= 512)
		{
			r = r + DirectIllumination(Ray(cam.origin + d * 40, normalize(d)), &s1, &s2)*(1. / samps);
		}
		else
		{
			r = r + RIS_DI(Ray(cam.origin + d * 40, normalize(d)), 32, &s1, &s2)*(1. / samps); 
		}
		//r = r + RIS_DI(Ray(cam.origin + d * 40, normalize(d)), 32, &s1, &s2)*(1. / samps); 
    }       // Camera rays are pushed ^^^^^ forward to start in interior   

	Colour fcolour;
	float3 colour = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
	//printf("colour: %f %f %f\n", colour.x, colour.y, colour.z);
	// convert from 96-bit to 24-bit colour + perform gamma correction
  	
	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255),
    (unsigned char)(powf(colour.y, 1 / 2.2f) * 255),
    (unsigned char)(powf(colour.z, 1 / 2.2f) * 255),1);
	
	//fcolour.components = make_uchar4((unsigned char)(55), (unsigned char)(25), (unsigned char)(255), 1);

	//printf("colour: %d %d %d\n", fcolour.components.x, fcolour.components.y, fcolour.components.z);
	finaloutputbuffer[i] = make_float3(x, y, fcolour.c);
	//finaloutputbuffer[i] = make_float3(colour.x, colour.y, colour.z);
	//printf("buffer: %f\n", finaloutputbuffer[i].x);
	//printf("buffer: %f\n", fcolour.c);
	// write rgb value of pixel to image buffer on the GPU, clamp value to [0.0f, 1.0f] range
    //output[i] = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
}

void render_gate(float3* finaloutputbuffer, float2 offset) {
	//float3* output_h = new float3[width*height]; // pointer to memory for image on the host (system RAM)
    //float3* output_d;    // pointer to memory for image on the device (GPU VRAM)

    // allocate memory on the CUDA device (GPU VRAM)
    //cudaMalloc(&output_d, width * height * sizeof(float3));
        
    // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
    dim3 block(16, 16, 1);   
    dim3 grid(scr_width / block.x, scr_height / block.y, 1);

    //printf("CUDA initialised.\nStart rendering...\n");
    
    // schedule threads on device and launch CUDA kernel from host
    render_kernel <<< grid, block >>>(finaloutputbuffer, offset);  

	// Wait for GPU to finish before accessing on host
  	cudaDeviceSynchronize();

    // copy results of computation from device back to host
    //cudaMemcpy(output_h, output_d, width * height *sizeof(float3), cudaMemcpyDeviceToHost);  
    
    // free CUDA memory
    //cudaFree(output_d);  

    //printf("Done!\n");

    // Write image to PPM file, a very simple image file format
    // FILE *f = fopen("smallptcuda.ppm", "w");          
    // fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    // for (int i = 0; i < width*height; i++)  // loop over pixels, write RGB values
    // fprintf(f, "%d %d %d ", ToInt(output_h[i].x),
    //                         ToInt(output_h[i].y),
    //                         ToInt(output_h[i].z));
    
    // printf("Saved image to 'smallptcuda.ppm'\n");

    //delete[] output_h;
    // system("PAUSE");
}