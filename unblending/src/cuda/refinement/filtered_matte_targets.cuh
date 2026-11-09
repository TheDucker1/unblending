#pragma once
#include "model/pixel_problem.cuh"
#include "guided_filter.cuh"

namespace general {
enum class NormalizationStatus:int {Valid=0,ZeroPlusSum=1,NonopaqueSourceOver=2};

HD inline void normalize_targets(int pixel,int n,int L,bool plus,const float* filtered,const float* first,float* targets,int* status){
    // The underlying filter uses a dummy opaque layer0 so EVERY real alpha,
    // including real background alpha, is treated as a filtered scalar channel.
    double sum=0;for(int l=0;l<L;++l){float a=configured::clip(filtered[size_t(l+1)*n+pixel]);targets[size_t(l)*n+pixel]=a;sum+=a;}
    NormalizationStatus code=NormalizationStatus::Valid;
    if(plus){if(sum>0)for(int l=0;l<L;++l)targets[size_t(l)*n+pixel]=float(double(targets[size_t(l)*n+pixel])/sum);else code=NormalizationStatus::ZeroPlusSum;}
    else if(std::abs(double(targets[pixel])-1.)>=1e-5)code=NormalizationStatus::NonopaqueSourceOver;
    for(int r=0;r<3;++r)targets[size_t(L+r)*n+pixel]=filtered[size_t(L+1+r)*n+pixel];
    status[pixel]=int(code);
}
__global__ void pack_filter(const float* first,float* packed,int n,int L){
    int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=n)return;packed[pixel]=1;
    for(int l=0;l<L;++l)packed[size_t(l+1)*n+pixel]=first[size_t(l)*n+pixel];
    for(int r=0;r<3;++r)packed[size_t(L+1+r)*n+pixel]=first[size_t(L+r)*n+pixel];
}
__global__ void finish_filter(const float* filtered,const float* first,float* target,int* status,int n,int L,bool plus){
    int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel<n)normalize_targets(pixel,n,L,plus,filtered,first,target,status);
}
template<class Model>bool normalization_kind(const Model* model,int L,bool opaque,bool smooth){
    if(smooth&&!opaque)throw std::invalid_argument("Original CPU smoothing requires an opaque background");
    bool plus=true,source_over=true;for(int l=0;l<L;++l){plus&=model[l].X==2&&model[l].Y==1&&model[l].Z==1;source_over&=model[l].X==1&&model[l].Y==1&&model[l].Z==1;}
    if(!plus&&!source_over)throw std::invalid_argument("Original CPU refinement normalization requires an all-Plus or all-SourceOver stack");
    return plus;
}

class FilterGPU {
    int n,L;bool plus;float *packed=nullptr,*filtered=nullptr;int* status=nullptr;
    twostage::FilterGPU core;
public:
    template<class Model>FilterGPU(int width,int height,int layers,bool opaque,bool smooth,const Model* model):n(width*height),L(layers),plus(normalization_kind(model,layers,opaque,smooth)),core(width,height,layers+1,smooth){
        twostage::cuda_check(cudaMalloc(&packed,size_t(L+4)*n*sizeof(float)));
        twostage::cuda_check(cudaMalloc(&filtered,size_t(L+4)*n*sizeof(float)));
        twostage::cuda_check(cudaMalloc(&status,size_t(n)*sizeof(int)));
    }
    ~FilterGPU(){cudaFree(packed);cudaFree(filtered);cudaFree(status);}
    const int* device_status()const{return status;}
    void run(const float* rgb,const float* first,float* targets,int radius,cudaStream_t stream=0){
        pack_filter<<<(n+127)/128,128,0,stream>>>(first,packed,n,L);
        core.run(rgb,packed,filtered,radius,stream);
        finish_filter<<<(n+127)/128,128,0,stream>>>(filtered,first,targets,status,n,L,plus);
        twostage::cuda_check(cudaGetLastError());
    }
};

class FilterCPU {
    int n,L;bool plus;twostage::Pool& pool;
    std::vector<float> packed,filtered;std::vector<int> status;
    twostage::FilterCPU core;
public:
    template<class Model>FilterCPU(int width,int height,int layers,bool opaque,bool smooth,const Model* model,twostage::Pool& workers):n(width*height),L(layers),plus(normalization_kind(model,layers,opaque,smooth)),pool(workers),packed(size_t(L+4)*n),filtered(packed.size()),status(n),core(width,height,layers+1,smooth,workers){}
    const std::vector<int>& statuses()const{return status;}
    void run(const float* rgb,const float* first,float* targets,int radius){
        pool.run(n,256,[&](int begin,int end){for(int p=begin;p<end;++p){packed[p]=1;for(int l=0;l<L;++l)packed[size_t(l+1)*n+p]=first[size_t(l)*n+p];for(int r=0;r<3;++r)packed[size_t(L+1+r)*n+p]=first[size_t(L+r)*n+p];}});
        core.run(rgb,packed.data(),filtered.data(),radius);
        pool.run(n,256,[&](int begin,int end){for(int p=begin;p<end;++p)normalize_targets(p,n,L,plus,filtered.data(),first,targets,status.data());});
    }
};
}
