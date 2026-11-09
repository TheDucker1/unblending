#pragma once
#include "pixel_problem.cuh"

namespace general {
// Device descriptors retain composition fields and carry an explicitly packed
// device pointer/count for variable-size model parameters. No host vtable or
// host pointer is copied into a kernel and called there.
struct PolynomialLayer:Layer {
    const double* terms=nullptr;
    int term_count=0;
};
struct PolynomialPrior {
    // D(c)=Gaussian(c)+sum_j weight_j*(dot(direction_j,c)-target_j)^4.
    // Each serialized term is [directionRGB,target,weight]. This example has
    // cross-channel non-Gaussian coupling and a runtime number of terms/layer.
    template<class T>HD static T evaluate(const PolynomialLayer& model,const T* color,T* gradient){
        T value=GaussianPrior::evaluate(model,color,gradient);
        for(int j=0;j<model.term_count;++j){const double* term=model.terms+5*j;T residual=-T(term[3]);
            for(int r=0;r<3;++r)residual+=T(term[r])*color[r];T square=residual*residual;
            value+=T(term[4])*square*square;
            if(gradient)for(int r=0;r<3;++r)gradient[r]+=T(4)*T(term[4])*square*residual*T(term[r]);
        }
        return value;
    }
};
}
