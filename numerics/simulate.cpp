// Exact age-envelope rejection sampler for the discrete ageing attachment tree.
// Compile with c++ -O3 -std=c++17 simulate.cpp -o simulate.
// Usage: ./simulate ALPHA_PERCENT REPLICATES FINAL_SIZE > results/tree_50.csv
#include "root_clock.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static double unit_uniform(std::mt19937_64& rng) {
    return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: simulate ALPHA_PERCENT REPLICATES FINAL_SIZE\n";
        return 1;
    }
    const int code = std::stoi(argv[1]), runs = std::stoi(argv[2]), limit = std::stoi(argv[3]);
    if (code < 0 || code >= 100 || runs < 1 || limit < 1000) return 2;
    const double alpha = code / 100.0;
    std::vector<double> weight(limit + 1), cumulative(limit + 1);
    for (int j = 1; j <= limit; ++j) {
        weight[j] = std::pow(j, -alpha);
        cumulative[j] = cumulative[j - 1] + weight[j];
    }
    std::vector<int> checkpoints;
    for (long long p = 1000; p <= limit; p *= 10) checkpoints.push_back(static_cast<int>(p));
    if (checkpoints.back() != limit) checkpoints.push_back(limit);

    const AgeKernel kernel(alpha, weight);
    std::cerr << "Largest relative integer-kernel error: " << kernel.maximum_relative_error << "\n";
    constexpr int bins = 20, kmax = 16;
    const std::vector<int> thresholds = {1,2,3,4,5,6,8,10,12,16,20,25,32,40,50,64,80,100,128,160,200,256,320,400,500,640,800,1000};
    std::cout << "alpha,replicate,seed,n,T_scaled,root_degree,proposals";
    for (int j = 0; j < bins; ++j) std::cout << ",cohort" << j+1;
    for (int k = 1; k <= kmax; ++k) std::cout << ",degree" << k;
    std::cout << ",root_A,root_Z,max_degree";
    for (int k : thresholds) std::cout << ",above" << k;
    std::cout << "\n" << std::setprecision(12);
    for (int replicate = 0; replicate < runs; ++replicate) {
        const std::uint64_t seed = 20261108ULL + 1000003ULL * code + 1009ULL * replicate;
        std::mt19937_64 rng(seed);
        AgeKernel::Path clock(kernel, weight[2]);
        std::vector<int> degree(limit + 1, 1), parent_at_step(limit + 1, 0);
        std::uint64_t proposals = 0;
        std::size_t checkpoint = 0;
        for (int n = 2; n < limit; ++n) {
            int parent = 0;
            while (!parent) {
                ++proposals;
                double target = unit_uniform(rng) * cumulative[n];
                int a = std::lower_bound(cumulative.begin()+1, cumulative.begin()+n+1, target)
                        - cumulative.begin();
                int slot = n-a+1;
                if ((rng() & 1ULL) == 0) {
                    parent = slot;
                } else if (slot >= 2 && slot < n) {
                    int candidate = parent_at_step[slot];
                    if (unit_uniform(rng) < weight[n-candidate+1]/weight[a]) parent = candidate;
                }
            }
            clock.next(n, parent, weight[n]);
            parent_at_step[n] = parent;
            ++degree[parent];
            const int m = n+1;
            if (m <= 256) {
                double direct=0.;
                for (int v=1; v<=m; ++v) direct+=degree[v]*weight[m-v+1];
                if (std::abs(clock.total/direct-1.)>2.1e-7)
                    throw std::logic_error("root clock at small size");
            }
            if (m != checkpoints[checkpoint]) continue;

            double total_weight = 0.0;
            long long degree_sum = 0;
            int maximum=*std::max_element(degree.begin()+1,degree.begin()+m+1);
            std::vector<long long> cohort(bins), freq(std::max(kmax,maximum)+1), counts(bins);
            for (int v = 1; v <= m; ++v) {
                total_weight += degree[v] * weight[m-v+1];
                degree_sum += degree[v];
                // Equal birth-time intervals (b/20, (b+1)/20].
                int bin = static_cast<int>((static_cast<long long>(bins)*v - 1) / m);
                cohort[bin] += degree[v];
                ++counts[bin];
                ++freq[degree[v]];
            }
            if (std::abs(clock.total/total_weight-1.)>2.1e-7)
                throw std::logic_error("root clock at checkpoint");
            if (degree_sum != 2LL*(m-1)) throw std::logic_error("degree sum");
            std::cout << alpha << ',' << replicate << ',' << seed << ',' << m << ','
                      << total_weight / std::pow(m,1-alpha) << ',' << degree[1] << ',' << proposals;
            for (int j = 0; j < bins; ++j) std::cout << ',' << double(cohort[j])/counts[j];
            for (int k = 1; k <= kmax; ++k) std::cout << ',' << double(freq[k])/m;
            const double product=std::exp(clock.log_root_product);
            std::cout << ',' << product << ',' << degree[1]/product << ',' << maximum;
            for (int k=maximum-1; k>=1; --k) freq[k]+=freq[k+1];
            for (int k : thresholds) std::cout << ',' << (k<maximum ? freq[k+1] : 0);
            std::cout << '\n';
            ++checkpoint;
        }
    }
}
