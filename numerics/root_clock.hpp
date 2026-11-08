// Positive exponential approximation used only for measuring the root martingale.
// Parents are sampled from the original power weights. All integer kernel values
// through the requested final size are checked against those weights before use.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

class AgeKernel {
    static constexpr int first = -64, last = 8, stride = 1024;
    static constexpr int count = last-first+2;
    using Vector = std::array<double, count>;
    Vector coefficient{}, rate{}, decay{};
    std::vector<Vector> remainder, quotient;
    static double attenuation(double t) { return t > 48.0 ? 0.0 : std::exp(-t); }

public:
    double maximum_relative_error = 0.0;
    const bool classical;
    AgeKernel(double alpha, const std::vector<double>& weight)
        : remainder(stride), quotient((weight.size()-1)/stride+1), classical(alpha==0.) {
        if (classical) return;
        // Laplace integral on a logarithmic grid, with the lower remainder at rate zero.
        constexpr double step = .5;
        double scale = step/std::tgamma(alpha);
        coefficient[0] = scale*std::exp(alpha*step*(first-1))/(-std::expm1(-alpha*step));
        for (int j=first; j<=last; ++j) {
            int i=j-first+1;
            rate[i]=std::exp(step*j);
            coefficient[i]=scale*std::exp(alpha*step*j-rate[i]);
        }
        for (int i=0; i<count; ++i) decay[i]=attenuation(rate[i]);
        for (int a=0; a<stride; ++a)
            for (int i=0; i<count; ++i) remainder[a][i]=attenuation(rate[i]*a);
        for (size_t q=0; q<quotient.size(); ++q)
            for (int i=0; i<count; ++i) quotient[q][i]=attenuation(rate[i]*stride*q);
        for (size_t age=1; age<weight.size(); ++age) {
            const auto& low=remainder[(age-1)%stride];
            const auto& high=quotient[(age-1)/stride];
            double estimate=0.;
            #pragma omp simd reduction(+:estimate)
            for (int i=0; i<count; ++i) estimate+=coefficient[i]*low[i]*high[i];
            maximum_relative_error=std::max(maximum_relative_error,std::abs(estimate/weight[age]-1.));
        }
        if (maximum_relative_error>2.e-7) throw std::runtime_error("age-kernel approximation out of tolerance");
    }

    class Path {
        const AgeKernel& kernel;
        Vector state{};
    public:
        double total, log_root_product=0.;
        Path(const AgeKernel& k, double weight_at_two): kernel(k), total(1.+weight_at_two) {
            for (int i=0; i<count; ++i) state[i]=1.+k.decay[i];
        }
        void next(int n, int parent, double root_weight) {
            log_root_product+=std::log1p(root_weight/total);
            if (kernel.classical) {
                total=2.*n;
                return;
            }
            int age=n-parent+1;
            const auto& low=kernel.remainder[age%stride];
            const auto& high=kernel.quotient[age/stride];
            double next_total=0.;
            #pragma omp simd reduction(+:next_total)
            for (int i=0; i<count; ++i) {
                state[i]=kernel.decay[i]*state[i]+1.+low[i]*high[i];
                next_total+=kernel.coefficient[i]*state[i];
            }
            total=next_total;
        }
    };
};
