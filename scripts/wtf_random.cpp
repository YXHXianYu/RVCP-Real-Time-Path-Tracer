#include <bits/stdc++.h>

float fract(float x) {
    return x - floor(x);
}

float g_rand_seed = 1.0;
float irand() {
    // g_rand_seed = fract(sin(g_rand_seed) * 43758.5453); // (len = 18)
    g_rand_seed = fract(sin(g_rand_seed) * 3758.5453); // (len = 2)
    return g_rand_seed;
}

float base_seed = 100.0;
int   index     = 0;

float xrand(float x) {
    return fract(sin(x) * 3758.5453); // (len = 2)
}
float yrand() {
    return xrand(base_seed + (++index));
}

int main() {
    for (int i = 0; i <= 1000; i++) {
        std::cout << yrand() << " ";
        if (i > 0 && i % 18 == 0) std::cout << std::endl;
    }
    return 0;
}