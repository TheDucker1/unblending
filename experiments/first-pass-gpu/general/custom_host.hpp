#pragma once
#include "fixture.hpp"
#include "custom_prior.cuh"
#include <algorithm>
#include <memory>

extern "C" void* original_custom_create(const double*,const double*,const double*,int);
extern "C" void original_custom_destroy(void*);
extern "C" double original_custom_evaluate(const void*,const double*,double*);

namespace custom_backend {
struct Model:general::PolynomialLayer {
    // Twice the inverse of a constant positive Hessian majorizer. Composition
    // uses the inherited Layer fields; the actual prior still uses S/terms.
    double covariance[9];
    double gray_inverse=0; // 1/(ones^T H_majorizer ones), restricted before inversion
};
struct HostModel:general::Layer {const void* object=nullptr;};
struct VirtualPrior {
    static double evaluate(const HostModel& m,const double* c,double* g){return original_custom_evaluate(m.object,c,g);}
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

class Input {
    struct Deleter {void operator()(void* p)const{original_custom_destroy(p);}};
    std::vector<std::unique_ptr<void,Deleter>> objects;
public:
    std::vector<double> terms;std::vector<Model> models;std::vector<HostModel> host_models;
    explicit Input(const Case& c,const std::string& path):models(c.p.layers),host_models(c.p.layers){
        if(c.custom)throw std::invalid_argument("Use an explicit typed payload instead of the legacy quartic tag");
        std::ifstream f(path,std::ios::binary);std::vector<int> counts(c.p.layers);read(f,counts.data(),counts.size());size_t n=0;
        for(int count:counts){if(count<0||count>100000)throw std::invalid_argument("Invalid term count");n+=size_t(count)*5;}
        if(n>size_t(64)*1024*1024/sizeof(double))throw std::invalid_argument("Custom example payload exceeds 64MiB contract");
        terms.resize(n);read(f,terms.data(),n);if(f.peek()!=EOF)throw std::invalid_argument("Trailing custom payload");for(double x:terms)if(!std::isfinite(x))throw std::invalid_argument("Nonfinite custom payload");
        size_t offset=0;
        for(int l=0;l<c.p.layers;++l){auto& m=models[l];static_cast<general::Layer&>(m)=c.model[l];for(double x:m.mean)if(!std::isfinite(x))throw std::invalid_argument("Nonfinite representative color");for(double x:m.inverse)if(!std::isfinite(x))throw std::invalid_argument("Nonfinite base precision");m.term_count=counts[l];m.terms=counts[l]?terms.data()+offset:nullptr;offset+=5*counts[l];metric(m);
            objects.emplace_back(original_custom_create(m.mean,m.inverse,m.terms,m.term_count));static_cast<general::Layer&>(host_models[l])=c.model[l];host_models[l].object=objects.back().get();}
    }
    Input(const Input&)=delete;Input& operator=(const Input&)=delete;
};

class Device {
    double* payload=nullptr;
public:
    Model* models=nullptr;
    explicit Device(const Input& host){
        try{if(!host.terms.empty()){ck(cudaMalloc(&payload,host.terms.size()*sizeof(double)));ck(cudaMemcpy(payload,host.terms.data(),host.terms.size()*sizeof(double),cudaMemcpyHostToDevice));}
            auto descriptors=host.models;size_t offset=0;for(auto& m:descriptors){m.terms=m.term_count?payload+offset:nullptr;offset+=5*m.term_count;}
            ck(cudaMalloc(&models,descriptors.size()*sizeof(Model)));ck(cudaMemcpy(models,descriptors.data(),descriptors.size()*sizeof(Model),cudaMemcpyHostToDevice));}
        catch(...){cudaFree(payload);cudaFree(models);throw;}
    }
    ~Device(){cudaFree(models);cudaFree(payload);}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
};
}
