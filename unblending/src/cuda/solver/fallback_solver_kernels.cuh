#pragma once
#include "augmented_lagrangian.cuh"
#include "refinement/streaming_guided_filter.cuh"
#include "solver/feasibility/jacobian_correction.cuh"
#include <climits>

namespace general {
template<class T,class Prior,class Model>__global__ void solve_tiles(const Model* model,Problem p,int n,int offset,int count,int workspace_stride,
    const T* initial,const T* targets,const T* alpha,const T* backgrounds,T* output,T* memory,Options options,Work* works,const int* input_status,int max_constraints,int* errors){
    int local=blockIdx.x*blockDim.x+threadIdx.x;if(local>=count)return;int pixel=offset+local;
    int error=p.constraints(model)>max_constraints?1:(input_status&&input_status[pixel]?100+input_status[pixel]:0);errors[pixel]=error;
    if(error){for(int k=0;k<4*p.layers;++k)output[size_t(k)*n+pixel]=initial[size_t(k)*n+pixel];works[pixel]=Work{};return;}
    T target[3],bg[3];for(int r=0;r<3;++r){target[r]=targets[size_t(r)*n+pixel];bg[r]=backgrounds?backgrounds[size_t(r)*n+pixel]:T(0);}
    Work work;Optimizer<T,Prior,Model>::solve(model,p,target,{alpha?alpha+pixel:nullptr,size_t(n)},bg,{initial+pixel,size_t(n)},
        {output+pixel,size_t(n)},{memory+local,size_t(workspace_stride)},options,work);works[pixel]=work;
}

template<class T,class Prior,class Model>__global__ void correct_tiles(const Model* model,Problem p,int n,int offset,int count,int stride,
    const T* targets,const T* alpha,const T* backgrounds,T* output,T* memory,Work* works,const int* errors){
    int local=blockIdx.x*blockDim.x+threadIdx.x;if(local>=count)return;int pixel=offset+local;if(errors[pixel])return;
    int N=4*p.layers,C=p.constraints(model);auto at=[&](int k){return View<T>{memory+local+size_t(k)*stride,size_t(stride)};};
    View<T> x=at(0),trial=at(N),prefix=at(2*N),jac=at(3*N),residual=at(4*N),lambda=at(4*N+C);
    T target[3],bg[3];for(int r=0;r<3;++r){target[r]=targets[size_t(r)*n+pixel];bg[r]=backgrounds?backgrounds[size_t(r)*n+pixel]:T(0);}
    for(int r=0;r<C;++r)lambda[r]=0;T energy;View<const T> original{output+pixel,size_t(n)},a{alpha?alpha+pixel:nullptr,size_t(n)};
    Program<T,Prior,Model>::evaluate(model,p,original,target,a,{lambda.data,lambda.stride},T(0),prefix,residual,jac,false,&energy);
    T norm=0;for(int r=0;r<C;++r)norm+=residual[r]*residual[r];unsigned calls=1;
    bool changed=Repair<T,Prior,Model>::run(model,p,target,a,bg,original,x,trial,prefix,jac,residual,lambda,energy,norm,&calls);
    if(changed)for(int k=0;k<N;++k)output[size_t(k)*n+pixel]=x[k];works[pixel].correction_values=calls;works[pixel].corrections=changed;
}

// Reusable in-memory GPU backend. Input/output planes retain full image stride;
// iterative state/history uses a bounded reusable pixel tile. This avoids
// allocating history_size*layers*all_image_pixels and has no CPU iteration or
// transfer between tiles. A CUDA graph may capture all tile launches.
template<class T,class Prior=GaussianPrior,class Model=Layer>class Workspace {
    int n,L,max_constraints,capacity;size_t bytes;Options options;
    T* memory=nullptr;Work* works=nullptr;int* errors=nullptr;
public:
    Workspace(int pixels,int layers,int constraints,Options opt,size_t budget=size_t(256)*1024*1024):n(pixels),L(layers),max_constraints(constraints),options(opt){
        if(n<1||L<1||constraints<1||opt.history<1||opt.history>64||opt.starts<1||opt.starts>3||opt.inner<1||opt.outer<1)throw std::invalid_argument("Invalid general GPU workspace configuration");
        size_t planes=size_t(8+2*opt.history)*4*L+2*opt.history+size_t(2)*constraints;
        if(planes>INT_MAX||L>INT_MAX/4)throw std::length_error("Per-pixel GPU workspace exceeds addressable solver state");
        size_t limit=budget/(planes*sizeof(T));if(!limit)throw std::length_error("Workspace budget cannot hold one pixel");
        capacity=int(limit<size_t(n)?limit:size_t(n));bytes=planes*capacity*sizeof(T);
        try{twostage::cuda_check(cudaMalloc(&memory,bytes));twostage::cuda_check(cudaMalloc(&works,size_t(n)*sizeof(Work)));twostage::cuda_check(cudaMalloc(&errors,size_t(n)*sizeof(int)));}
        catch(...){cudaFree(memory);cudaFree(works);cudaFree(errors);throw;}
    }
    Workspace(const Workspace&)=delete;Workspace& operator=(const Workspace&)=delete;
    ~Workspace(){cudaFree(memory);cudaFree(works);cudaFree(errors);}
    size_t allocated_bytes()const{return bytes;}
    int tile_capacity()const{return capacity;}
    const Work* device_work()const{return works;}
    const int* device_errors()const{return errors;}
    void solve(const Model* device_models,Problem problem,const T* rgb,const T* alpha,const T* background,
        const T* initial,T* output,const int* input_status=nullptr,cudaStream_t stream=0){
        if(problem.layers!=L)throw std::invalid_argument("Workspace layer count mismatch");
        if(problem.refinement&&(!alpha||(problem.smooth&&!background)))throw std::invalid_argument("Missing refinement alpha/background targets");
        if(3+(problem.refinement?L:1)>max_constraints)throw std::invalid_argument("Workspace constraint capacity mismatch");
        // Constraints are known from the host descriptor when constructing this
        // object. No device model dereference or GPU-to-CPU copy occurs here.
        for(int offset=0;offset<n;offset+=capacity){int count=n-offset<capacity?n-offset:capacity;
            solve_tiles<T,Prior,Model><<<(count+127)/128,128,0,stream>>>(device_models,problem,n,offset,count,capacity,initial,rgb,alpha,background,output,memory,options,works,input_status,max_constraints,errors);
            if(options.correction)correct_tiles<T,Prior,Model><<<(count+127)/128,128,0,stream>>>(device_models,problem,n,offset,count,capacity,rgb,alpha,background,output,memory,works,errors);}
        twostage::cuda_check(cudaGetLastError());
    }
};

template<class T,class Model>__global__ void initialize_first(const Model* model,Problem p,int n,T* layers){
    int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=n)return;
    for(int l=0;l<p.layers;++l){layers[size_t(l)*n+pixel]=p.opaque&&l==0?T(1):T(.5);for(int r=0;r<3;++r)layers[size_t(p.layers+3*l+r)*n+pixel]=configured::clip(T(model[l].mean[r]));}
}
template<class T>__global__ void initialize_refinement(Problem p,int n,const T* first,const T* targets,T* layers){
    int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=n)return;int L=p.layers;
    for(int l=0;l<L;++l)layers[size_t(l)*n+pixel]=p.opaque&&l==0?T(1):targets[size_t(l)*n+pixel];
    for(int k=L;k<4*L;++k)layers[size_t(k)*n+pixel]=p.smooth&&k<L+3?targets[size_t(k)*n+pixel]:first[size_t(k)*n+pixel];
}

// Both stages, all smoothing/normalization and all iterative work stay on the
// device. Host descriptors are used ONLY for immutable configuration metadata;
// custom payloads must already have their device descriptors prepared.
template<class T,class Prior=GaussianPrior,class Model=Layer>class Pipeline {
    int n,L,radius;Problem first_problem,second_problem;
    Workspace<T,Prior,Model> workspace;
    StreamingFilterGPU<T> filter;
    T *first=nullptr,*final=nullptr,*targets=nullptr;
    static int maximum_constraints(const Model* model,int layers){Problem p;p.layers=layers;p.refinement=true;return p.constraints(model);}
    static int checked_pixels(int w,int h){if(w<1||h<1||int64_t(w)*h>INT_MAX)throw std::invalid_argument("Invalid pipeline image dimensions");return w*h;}
public:
    Pipeline(int width,int height,const Model* host_models,int layers,bool opaque,bool smooth,Options opt,
        bool fixed_alpha=false,bool sparsity=false,size_t budget=size_t(256)*1024*1024):n(checked_pixels(width,height)),L(layers),radius(60*(width<height?width:height)/1000),
        workspace(n,layers,maximum_constraints(host_models,layers),opt,budget),filter(width,height,layers,opaque,smooth,host_models){
        first_problem.layers=L;first_problem.opaque=opaque;first_problem.sparsity=sparsity;
        second_problem=first_problem;second_problem.refinement=true;second_problem.minimum=false;second_problem.sparsity=false;second_problem.smooth=smooth;second_problem.fixed_alpha=fixed_alpha;
        try{twostage::cuda_check(cudaMalloc(&first,size_t(4)*L*n*sizeof(T)));twostage::cuda_check(cudaMalloc(&final,size_t(4)*L*n*sizeof(T)));twostage::cuda_check(cudaMalloc(&targets,size_t(L+3)*n*sizeof(T)));}
        catch(...){cudaFree(first);cudaFree(final);cudaFree(targets);throw;}
    }
    Pipeline(const Pipeline&)=delete;Pipeline& operator=(const Pipeline&)=delete;
    ~Pipeline(){cudaFree(first);cudaFree(final);cudaFree(targets);}
    const T* first_layers()const{return first;}
    const T* final_layers()const{return final;}
    const T* filtered_targets()const{return targets;}
    const int* normalization_status()const{return filter.device_status();}
    const int* solver_errors()const{return workspace.device_errors();}
    size_t workspace_bytes()const{return workspace.allocated_bytes();}
    int tile_capacity()const{return workspace.tile_capacity();}
    void first_pass(const Model* models,const T* rgb,cudaStream_t stream=0){
        initialize_first<<<(n+127)/128,128,0,stream>>>(models,first_problem,n,first);
        workspace.solve(models,first_problem,rgb,nullptr,nullptr,first,first,nullptr,stream);
    }
    void refinement(const Model* models,const T* rgb,const T* initial,cudaStream_t stream=0){
        filter.run(rgb,initial,targets,radius,stream);initialize_refinement<<<(n+127)/128,128,0,stream>>>(second_problem,n,initial,targets,final);
        workspace.solve(models,second_problem,rgb,targets,targets+size_t(L)*n,final,final,filter.device_status(),stream);
    }
    void run(const Model* models,const T* rgb,cudaStream_t stream=0){first_pass(models,rgb,stream);refinement(models,rgb,first,stream);}
};
}
