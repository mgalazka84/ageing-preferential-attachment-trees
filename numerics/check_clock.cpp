// Compare every step and the accumulated root product against direct degree sums.
#include "root_clock.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    constexpr int limit=1024;
    for (double alpha : {0., .25, .5, .75, .9}) {
        std::vector<double> weight(limit+1);
        for (int j=1; j<=limit; ++j) weight[j]=std::pow(j,-alpha);
        AgeKernel kernel(alpha,weight);
        for (int history=0; history<3; ++history) {
            AgeKernel::Path clock(kernel,weight[2]);
            std::vector<int> degree(limit+1,1);
            double log_product=0.;
            for (int n=2; n<limit; ++n) {
                double total=0.;
                for (int v=1; v<=n; ++v) total+=degree[v]*weight[n-v+1];
                if (std::abs(clock.total/total-1.)>2.1e-7)
                    throw std::logic_error("total differs from direct sum");
                log_product+=std::log1p(weight[n]/total);
                int parent=1;
                if (history==1) parent=n;
                if (history==2) parent=1+(37*n+17)%(n-1);
                clock.next(n,parent,weight[n]);
                ++degree[parent];
                if (std::abs(clock.log_root_product-log_product)>1.5e-6)
                    throw std::logic_error("root product differs from direct evaluation");
            }
        }
    }
    std::cout << "Direct root-clock and product checks passed on 15 deterministic histories.\n";
}
