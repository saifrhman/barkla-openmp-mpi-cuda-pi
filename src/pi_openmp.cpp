#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

int main(int argc, char** argv) {
    // Number of samples (default = 10 million)
    long long N = 10000000;

    if (argc > 1) {
        N = std::stoll(argv[1]);
    }

    std::cout << "Samples: " << N << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    long long inside = 0;

    #pragma omp parallel
    {
        std::mt19937_64 rng(1234 + omp_get_thread_num());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // parallel loop with reduction on inside
        #pragma omp for reduction(+:inside)
        for (long long i = 0; i < N; ++i) {
            double x = dist(rng);
            double y = dist(rng);
            if (x * x + y * y <= 1.0) {
                inside++;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    double pi_est = 4.0 * (double)inside / (double)N;
    std::cout << "Estimated pi = " << pi_est << std::endl;
    std::cout << "Elapsed time = " << elapsed << " s" << std::endl;

    return 0;
}
