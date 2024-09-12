/**
 * Run this script and you'll find something interesting (in stdout).
 */

#include <bits/stdc++.h>

struct vec3 {
    double x, y, z;
    vec3() : x(0), y(0), z(0) {}
    vec3(double x) : x(x), y(x), z(x) {}
    vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    vec3 operator-() const { return vec3(-x, -y, -z); }
    vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
    vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
    vec3 operator*(double v) const { return vec3(x * v, y * v, z * v); }
    vec3 operator/(double v) const { return vec3(x / v, y / v, z / v); }
};

double dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double rand_unit() {
    return double(rand()) / RAND_MAX;
}

vec3 normalize(const vec3& v) {
    return v / sqrt(dot(v, v));
}

vec3 rand3() {
    return vec3(rand_unit(), rand_unit(), rand_unit());
}

vec3 random_in_unit_sphere() { // reject method
    vec3 p;
    do {
        p = rand3() * 2.0 - vec3(1.0);
    } while (dot(p, p) >= 1.0);
    return p;
}

// #define PI 3.1415926

// vec3 random_in_unit_sphere() {
//     float theta = 2 * PI * rand_unit();
//     float phi   = acos(1 - 2 * rand_unit());

//     float x = sin(phi) * cos(theta);
//     float y = sin(phi) * sin(theta);
//     float z = cos(phi);

//     return vec3(x, y, z);
// }

vec3 random_in_unit_sphere_surface() {
    return normalize(random_in_unit_sphere());
}

vec3 random_in_unit_hemisphere(vec3 normal) {
    vec3 in_unit_sphere = random_in_unit_sphere();
    return dot(in_unit_sphere, normal) > 0.0 ? in_unit_sphere : -in_unit_sphere;
}

vec3 random_in_unit_hemisphere_surface(vec3 normal) {
    // return normalize(random_in_unit_hemisphere(normal));
    vec3 in_unit_sphere_surface = random_in_unit_sphere_surface();
    return dot(in_unit_sphere_surface, normal) > 0.0 ? in_unit_sphere_surface : -in_unit_sphere_surface;
}

using uint = uint32_t;

int main() {
    srand(time(0));

    const uint   SUM   = 10000000;
    const double X_MIN = -0.1, X_MAX = 0.1;
    const double Y_MIN = -0.1, Y_MAX = 0.1;
    const double Z_MIN = 0, Z_MAX = 0.1;

    uint cntA[10] = {};
    uint cntB[10] = {};

    vec3 normal = vec3(0, 1, 0);
    for (uint i = 0; i < SUM; i++) {
        {
            vec3 v = normalize(random_in_unit_hemisphere_surface(normal));
            cntA[int(dot(v, normal) * 10)]++;
        }

        {
            vec3 v = normalize(normal + random_in_unit_sphere_surface());
            cntB[int(dot(v, normal) * 10)]++;
        }
    }

    for (int i = 0; i < 10; i++) {
        std::cout << cntA[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < 10; i++) {
        std::cout << cntB[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}