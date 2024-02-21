#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <random>

#define INV_PI 0.31830988618

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0, 1);

double randomNumber() // Random number from 0 to 1
{
	return dis(gen);
}

struct Vector3f 
{
	float x, y, z;
	Vector3f(double x_ = 0, double y_ = 0, double z_ = 0) { x = x_; y = y_; z = z_; }
	Vector3f operator+(const Vector3f& v) const {
		return Vector3f(x + v.x, y + v.y, z + v.z);
	}
	Vector3f operator-(const Vector3f& v) const {
		return Vector3f(x - v.x, y - v.y, z - v.z);
	}
	Vector3f operator*(double f) const {
		return Vector3f(x * f, y * f, z * f);
	}
	Vector3f Mult(const Vector3f& v) const {
		return Vector3f(x * v.x, y * v.y, z * v.z);
	}
	double Dot(const Vector3f& v) const {
		return x * v.x + y * v.y + z * v.z;
	}
	Vector3f& Normalize() {
		return *this = *this * (1 / sqrtf(x * x + y * y + z * z));
	}
	double Length() const {
		return sqrtf(x * x + y * y + z * z);
	}
	Vector3f operator%(Vector3f& b) 
	{ 
		return Vector3f(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); 
	}
};

struct Ray 
{
	Vector3f origin, direction;
	Ray(const Vector3f& m_origin, const Vector3f& m_direction) : origin(m_origin), direction(m_direction) {}
};

struct Sphere 
{
	double radius;
	Vector3f position, color;
	Sphere(double m_radius, const Vector3f& m_position, const Vector3f& m_color) : radius(m_radius), position(m_position), color(m_color) {}
	double Intersect(const Ray& ray) const {
		Vector3f op = position - ray.origin;
		double t, epsilon = 1e-4;
		double b = op.Dot(ray.direction), det = b * b - op.Dot(op) + radius * radius;
		if (det < 0) return 0;	else det = sqrt(det);
		return (t = b - det) > epsilon ? t : ((t = b + det) > epsilon ? t : 0);
	}
};

struct PointLight 
{
	Vector3f position, emission;
	PointLight(const Vector3f& m_position, const Vector3f& m_emission) : position(m_position), emission(m_emission) {}
};

Sphere spheres[] = {//Scene: radius, position, color
	Sphere(1e5, Vector3f(1e5 + 1,40.8,81.6), Vector3f(.75,.25,.25)),//Left
	Sphere(1e5, Vector3f(-1e5 + 99,40.8,81.6), Vector3f(.25,.25,.75)),//Rght
	Sphere(1e5, Vector3f(50,40.8, 1e5), Vector3f(.75,.75,.75)),//Back
	Sphere(1e5, Vector3f(50,40.8, -1e5 + 170), Vector3f()),//Frnt
	Sphere(1e5, Vector3f(50, 1e5, 81.6), Vector3f(.75,.75,.75)),//Botm
	//Sphere(1e5, Vector3f(50,1e5 + 81.6,81.6), Vector3f(.75,.75,.75)),//Top
	Sphere(16.5,Vector3f(27,16.5,47), Vector3f(.75,.55,.25) * .999),//Ball1
	Sphere(16.5,Vector3f(73,16.5,78), Vector3f(.55,.25,.75) * .999),//Ball2
};

PointLight lights[] = {// Scene: positioin, emission
	PointLight(Vector3f(50, 81.6 - 1.27, 76.6), Vector3f(5000, 5000, 5000)),
	PointLight(Vector3f(40, 81.6 - 2.27, 26.6), Vector3f(5000, 5000, 5000)),
	PointLight(Vector3f(60, 81.6 - 2.27, 106.6), Vector3f(5000, 5000, 5000)),
	PointLight(Vector3f(50, 81.6 - 2.27, 76.6), Vector3f(5000, 5000, 5000))
};

inline double Clamp(double x) 
{
	return x < 0 ? 0 : x > 1 ? 1 : x;
}

inline int ToInt(double x) 
{
	return int(pow(Clamp(x), 1 / 2.2) * 255 + .5);
}

inline bool Intersect(const Ray& ray, double& t, int& id) 
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

	void addSample(int lightID, float weight)
	{
		wSum = wSum + weight;
		if (randomNumber() < weight / wSum)
		{
			id = lightID;
		}
	}
};

Vector3f RIS_DI(const Ray& r, const int& M)
{
	double t; // distance to intersection
	int id = 0; // id of intersected object
	Ray ray = r;

	if (!Intersect(ray, t, id)) return Vector3f();
	const Sphere& obj = spheres[id]; // the hit object
	Vector3f hitPoint = ray.origin + ray.direction * t;
	Vector3f normal = (hitPoint - obj.position).Normalize();
	Vector3f normal_local = normal.Dot(ray.direction) < 0 ? normal : normal * -1;

	int lightsCount = sizeof(lights) / sizeof(PointLight);
	int RISSamples = std::min(lightsCount, M);

	Reservoir reservoir;

	for (int i = 0; i < RISSamples; i++)
	{
		// Pick a random light from the scene to sample
		int lightToSample = std::min(int(randomNumber() * lightsCount), lightsCount - 1);

		// Sample the light
		double distanceToLight = (lights[lightToSample].position - hitPoint).Length();
		Vector3f lightEmission = lights[lightToSample].emission;
		Vector3f lightDirection = (lights[lightToSample].position - hitPoint).Normalize();

		// Compute the Lambertian cosine
		double cosTheta = normal_local.Dot(lightDirection);
		if (cosTheta < 0) cosTheta = 0.;

		// Calculate the light attenuation
		double lightAttenuation = 1 / (distanceToLight * distanceToLight);

		// Compute the BRDF
		Vector3f BRDF = obj.color * INV_PI * cosTheta;

		// Compute the light intensity
		Vector3f lightIntensity = lightEmission * lightAttenuation;

		// Compute radiance
		Vector3f radiance = BRDF.Mult(lightIntensity);

		// Compute pHat
		float pHat = radiance.Length();

		// MIS weight
		float MISWeight = 1. / RISSamples;

		// Contribution weight of light
		float contributionWeight = lightsCount; // The inverse PDF of sampling the light

		// The weight of the sample
		float weight = pHat * MISWeight * contributionWeight;

		// Add the sample to the reservoir
		reservoir.addSample(lightToSample, weight);
	}

	// The chosen light to sample and the contribution weight
	int finalLight = reservoir.id;
	//printf("Final light: %d\n", finalLight);

	// Sample the light
	double distanceToLight = (lights[finalLight].position - hitPoint).Length();
	Vector3f lightEmission = lights[finalLight].emission;
	Vector3f lightDirection = (lights[finalLight].position - hitPoint).Normalize();
	
	// Compute the Lambertian cosine
	double cosTheta = normal_local.Dot(lightDirection);
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
	Vector3f BRDF = obj.color * INV_PI * cosTheta;

	// Compute the light intensity
	Vector3f lightIntensity = lightEmission * lightAttenuation;

	// Compute radiance
	Vector3f radiance = BRDF.Mult(lightIntensity) * visibility;

	// Compute pHat
	float pHat = radiance.Length();

	// pHat multiplied by the visibility
	float weight = pHat > 0.0 ? (1. / pHat) * reservoir.wSum : 0.0;

	// Compute the direct illumination of Lambertian BRDF
	Vector3f color = radiance * weight;

	return color;
}

Vector3f DirectIllumination(const Ray& r)
{
	double t; // distance to intersection
	int id = 0; // id of intersected object
	Ray ray = r;
	
	if (!Intersect(ray, t, id)) return Vector3f();
	const Sphere& obj = spheres[id]; // the hit object
	Vector3f hitPoint = ray.origin + ray.direction * t;
	Vector3f normal = (hitPoint - obj.position).Normalize();
	Vector3f normal_local = normal.Dot(ray.direction) < 0 ? normal : normal * -1;

	// Pick a random light from the scene to sample
	int lightsCount = sizeof(lights) / sizeof(PointLight);
	int lightToSample = std::min(int(randomNumber() * lightsCount), lightsCount - 1);

	// Sample the light
	double distanceToLight = (lights[lightToSample].position - hitPoint).Length();
	Vector3f lightEmission = lights[lightToSample].emission;
	Vector3f lightDirection = (lights[lightToSample].position - hitPoint).Normalize();

	// Compute the Lambertian cosine
	double cosTheta = normal_local.Dot(lightDirection);
	if (cosTheta < 0) return Vector3f();

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
	Vector3f shadingColor = obj.color.Mult(lightEmission) * INV_PI * cosTheta * lightAttenuation * visibility * invPdf;

	return shadingColor;
}

int main(int argc, char* argv[]) {
	int w = 1024, h = 768, samps = 1; // # samples
	Ray cam(Vector3f(50, 52, 295.6), Vector3f(0, -0.042612, -1).Normalize()); // cam pos, dir
	Vector3f cx = Vector3f(w * .5135 / h), cy = (cx % cam.direction).Normalize() * .5135, r, * c = new Vector3f[w * h];
#pragma omp parallel for schedule(dynamic, 1) private(r)       // OpenMP
	for (int y = 0; y < h; y++) {                       // Loop over image rows
		fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps, 100. * y / (h - 1));
		for (unsigned short x = 0; x < w; x++)   // Loop cols
		{
			int i = (h - y - 1) * w + x;
			Vector3f r = Vector3f();
			for (int s = 0; s < samps; s++) {
				Vector3f d = cx * ((.5 + x) / w - .5) +
					cy * ((.5 + y) / h - .5) + cam.direction;
				//r = r + DirectIllumination(Ray(cam.origin + d * 140, d.Normalize())) * (1. / samps);
				r = r + RIS_DI(Ray(cam.origin + d * 140, d.Normalize()), 32) * (1. / samps);
			} // Camera rays are pushed ^^^^^ forward to start in interior
			c[i] = c[i] + Vector3f(Clamp(r.x), Clamp(r.y), Clamp(r.z));
		}
	}
	FILE* f = fopen("image.ppm", "w");         // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for (int i = 0; i < w * h; i++)
		fprintf(f, "%d %d %d ", ToInt(c[i].x), ToInt(c[i].y), ToInt(c[i].z));
}