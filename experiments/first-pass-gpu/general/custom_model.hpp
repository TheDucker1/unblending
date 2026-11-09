#pragma once
#include "cuda_check.hpp"
#include "custom_prior.cuh"
#include <algorithm>

namespace custom_backend {
struct Model:general::PolynomialLayer {
    // Twice the inverse of a constant positive Hessian majorizer. Composition
    // uses the inherited Layer fields; the actual prior still uses S/terms.
    double covariance[9];
    double gray_inverse=0; // 1/(ones^T H_majorizer ones), restricted before inversion
};

inline void metric(Model& m){
    double h[3][3],a[3][6]{};
    for(int r=0;r<3;++r)for(int s=0;s<3;++s)h[r][s]=m.inverse[3*r+s]+m.inverse[3*s+r];
    // For every c in [0,1]^3, Hessian(D(c)) <= H in PSD order. Absolute
    // weights retain an upper majorizer even for a nonconvex signed term.
    for(int j=0;j<m.term_count;++j){const double* t=m.terms+5*j;double lo=-t[3],hi=lo;
        for(int r=0;r<3;++r){if(t[r]<0)lo+=t[r];else hi+=t[r];}
        double bound=std::max(std::abs(lo),std::abs(hi)),factor=12*std::abs(t[4])*bound*bound;
        for(int r=0;r<3;++r)for(int s=0;s<3;++s)h[r][s]+=factor*t[r]*t[s];}
    double scale=std::max({std::abs(h[0][0]),std::abs(h[1][1]),std::abs(h[2][2])});
    if(!(scale>0&&std::isfinite(scale)))throw std::invalid_argument("Nonfinite custom metric");
    double gray=0;for(int r=0;r<3;++r)for(int s=0;s<3;++s)gray+=h[r][s];
    m.gray_inverse=gray>0&&std::isfinite(gray)?1/gray:1/(9*scale);
    // Normalize before inversion; descriptor setup is outside both timers.
    for(int r=0;r<3;++r){for(int s=0;s<3;++s)a[r][s]=h[r][s]/scale;a[r][3+r]=1;}
    double determinant=a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
    if(!(a[0][0]>0&&a[0][0]*a[1][1]-a[0][1]*a[1][0]>0&&determinant>0))throw std::invalid_argument("Custom metric must be positive definite");
    for(int k=0;k<3;++k){int pivot=k;for(int r=k+1;r<3;++r)if(std::abs(a[r][k])>std::abs(a[pivot][k]))pivot=r;
        if(!(std::abs(a[pivot][k])>1e-15))throw std::invalid_argument("Ill-conditioned custom metric");
        if(pivot!=k)for(int s=0;s<6;++s)std::swap(a[k][s],a[pivot][s]);double d=a[k][k];for(int s=0;s<6;++s)a[k][s]/=d;
        for(int r=0;r<3;++r)if(r!=k){double v=a[r][k];for(int s=0;s<6;++s)a[r][s]-=v*a[k][s];}}
    for(int r=0;r<3;++r)for(int s=0;s<3;++s){m.covariance[3*r+s]=(a[r][3+s]+a[s][3+r])/scale;if(!std::isfinite(m.covariance[3*r+s]))throw std::invalid_argument("Nonfinite custom inverse metric");}
}
}
