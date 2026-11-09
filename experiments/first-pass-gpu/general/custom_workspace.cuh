#pragma once
#include "trace.cuh"
#include <memory>
#include "custom_refinement.cuh"
#include "device_backend.cuh"
#include "constraint_restore.cuh"
#include "reduced_gpu3.cuh"
#include "cube_api.cuh"
#include "register_first_gpu.cuh"

namespace custom_backend {
using general::View;
using Prior=general::PolynomialPrior;

// Original-double hard-constraint measurement, with no CPU reference point.
// The custom prior's score is used in the optimizer; this guard checks RGBA,
// gray equalities, bounds and the requested fixed-input contracts.
HD inline double constraint_norm(const Model* model,general::Problem p,View<const double> target,View<const double> alpha,View<const double> bg,View<const double> x){
    int L=p.layers;double rgb[3],da=x[0],error=0;if(p.opaque&&x[0]!=1)return INFINITY;
    for(int l=0;l<L;++l){double a=x[l];if(!std::isfinite(a)||a<0||a>1)return INFINITY;if(p.refinement){double d=a-alpha[l];error+=d*d;if(p.fixed_alpha&&d!=0)return INFINITY;}
        double norm=0;for(int r=0;r<3;++r){double c=x[L+3*l+r];if(!std::isfinite(c)||c<0||c>1)return INFINITY;norm+=c*c;if(p.refinement&&p.smooth&&l==0&&c!=bg[r])return INFINITY;}
        if(model[l].gray){norm=std::sqrt(norm);for(int r=0;r<3;++r){double d=1.7320508075688772935*x[L+3*l+r]-norm;error+=d*d;}}}
    for(int r=0;r<3;++r)rgb[r]=x[L+r];
    for(int l=1;l<L;++l){const auto& m=model[l];double a=x[l],next=double(m.X)*a*da+double(m.Y)*a*(1-da)+double(m.Z)*da*(1-a);for(int r=0;r<3;++r)rgb[r]=range_union::native_color(m,a,da,next,x[L+3*l+r],rgb[r]);da=next;}
    for(int r=0;r<3;++r){double d=rgb[r]-target[r];error+=d*d;}if(!p.refinement){double d=da-1;error+=d*d;}
    return std::isfinite(error)?error:INFINITY;
}

__global__ void cast(size_t count,const double* input,float* output){for(size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;k<count;k+=size_t(gridDim.x)*blockDim.x)output[k]=float(input[k]);}
__global__ void guard(const Model* model,general::Problem p,int n,const float* input,const double* target,const double* alpha,const double* bg,double* output,const int* raw,int* status,int* flags,int* list,int* count,double tolerance,const int* input_status,const int* configuration_class=nullptr){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;for(int k=0;k<4*p.layers;++k)output[size_t(k)*n+i]=input[size_t(k)*n+i];
    if(p.refinement&&(p.fixed_alpha||raw))for(int l=0;l<p.layers;++l)output[size_t(l)*n+i]=alpha[size_t(l)*n+i];
    if(p.refinement&&p.smooth)for(int r=0;r<3;++r)output[size_t(p.layers+r)*n+i]=bg[size_t(r)*n+i];
    if(input_status&&input_status[i]){status[i]=8;flags[i]=0;return;}
    if(configuration_class&&*configuration_class<0){status[i]=8;flags[i]=0;return;}
    auto cv=[&](const double* x){return View<const double>{x+i,size_t(n)};};double norm=constraint_norm(model,p,cv(target),cv(alpha),cv(bg),cv(output));
    bool bad=(raw&&raw[i]!=0&&raw[i]!=5)||norm>tolerance*tolerance;flags[i]=bad;status[i]=bad?3:0;if(bad)list[atomicAdd(count,1)]=i;
}

#ifdef __CUDACC__
#define CUSTOM_OPT_OUTLINED __noinline__
#else
#define CUSTOM_OPT_OUTLINED
#endif
template<class T>HD CUSTOM_OPT_OUTLINED void optimize(const Model* models,general::Problem p,const T* target,View<const T> alpha,const T* bg,View<const T> initial,View<T> output,View<T> memory,general::Options opt,general::Work& work){
    general::Optimizer<T,Prior,Model>::solve(models,p,target,alpha,bg,initial,output,memory,opt,work);
}
#undef CUSTOM_OPT_OUTLINED
template<class T>HD void generic_solve(const Model* models,general::Problem p,const T* target,View<const T> alpha,const T* bg,View<const T> initial,View<T> output,View<T> memory,general::Options opt,general::Work& work){
    optimize(models,p,target,alpha,bg,initial,output,memory,opt,work);
    if(!opt.correction)return;int N=4*p.layers,C=p.constraints(models);auto at=[&](int offset){return View<T>{memory.data+size_t(offset)*memory.stride,memory.stride};};
    auto x=at(0),trial=at(N),prefix=at(2*N),jac=at(3*N),residual=at(4*N),lambda=at(4*N+C);for(int r=0;r<C;++r)lambda[r]=0;T energy;
    general::Program<T,Prior,Model>::evaluate(models,p,{output.data,output.stride},target,alpha,{lambda.data,lambda.stride},T(0),prefix,residual,jac,false,&energy);T norm=0;for(int r=0;r<C;++r)norm+=residual[r]*residual[r];unsigned calls=1;
    bool improved=general::Repair<T,Prior,Model>::run(models,p,target,alpha,bg,{output.data,output.stride},x,trial,prefix,jac,residual,lambda,energy,norm,&calls);if(improved)for(int k=0;k<N;++k)output[k]=x[k];work.correction_values+=calls;work.corrections+=improved;
    using Restore=general::ConstraintRestore<T,Model>;
    if(Restore::norm(models,p,target,alpha,{output.data,output.stride})>T(opt.outer_tolerance*opt.outer_tolerance)){
        bool restored=Restore::run(models,p,target,alpha,bg,initial,output,memory,T(opt.outer_tolerance));work.corrections+=restored;
        // A feasible restored point is the new incumbent. Optimize the actual
        // prior from it, preserving it if trial steps cannot improve feasibly.
        if(restored&&Restore::norm(models,p,target,alpha,{output.data,output.stride})<=T(opt.outer_tolerance*opt.outer_tolerance)){
            general::Options polish=opt;polish.starts=1;polish.outer=6;polish.inner=std::min(opt.inner,128);
            optimize(models,p,target,alpha,bg,View<const T>{output.data,output.stride},output,memory,polish,work);
        }
    }
}

__global__ void repair(const Model* model,general::Problem p,int n,int slots,const double* initial,const double* target,const double* alpha,const double* bg,double* output,double* memory,size_t state_words,int state_tile,const int* list,const int* count,int* status,int* generic_flags,int* coupled_flags,general::Work* works,double tolerance,int rounds,bool fast,bool inverse,general::Options opt){
    int active=state_tile&&*count<slots?*count:slots;int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=active)return;View<double> work=custom_refinement::state_view(memory,slot,active,state_words,state_tile);
    for(int job=slot;job<*count;job+=active){int i=list[job];auto cv=[&](const double* x){return View<const double>{x+i,size_t(n)};};View<double> point{output+i,size_t(n)};int code=8,a=0,v=0;general::Work stats;
        if(fast&&inverse){if(p.refinement)code=custom_refinement::improve(model,p,cv(initial),cv(target),cv(alpha),cv(bg),point,work,tolerance,rounds,a,v);else code=custom_refinement::first(model,p,cv(initial),cv(target),cv(bg),point,work,tolerance,rounds,a,v);}
        if(fast&&code==8){code=coupled_sqp::Solver<double>::run(model,p,cv(initial),cv(target),cv(alpha),cv(bg),point,work,tolerance,rounds,a,v);coupled_flags[i]=1;}
        double norm=constraint_norm(model,p,cv(target),cv(alpha),cv(bg),cv(output));bool generic=!fast||(code!=0&&code!=5)||norm>tolerance*tolerance;generic_flags[i]=generic;
        if(generic){double t[3],b[3];for(int r=0;r<3;++r){t[r]=target[size_t(r)*n+i];b[r]=bg[size_t(r)*n+i];}generic_solve(model,p,t,cv(alpha),b,cv(initial),point,work,opt,stats);norm=constraint_norm(model,p,cv(target),cv(alpha),cv(bg),cv(output));}
        status[i]=norm<=tolerance*tolerance?0:3;works[i]=stats;
    }
}

class Workspace {
    int n,L,C,slots32,slots64,state_tile,cube_slots=0;bool fast,inverse;general::Options options;size_t words32,words64,bytes32,bytes64;
    custom_refinement::Graph graph;
    std::unique_ptr<reduced_gpu::Plan> reduced;
    register_first::Model<float>* cached_models=nullptr;int* cache_class=nullptr;
    general::Workspace<float,Prior,Model> general32;
    float *x32=nullptr,*rgb32=nullptr,*a32=nullptr,*bg32=nullptr,*out32=nullptr,*work32=nullptr;
    double* work64=nullptr;int *raw=nullptr,*statuses=nullptr,*flags=nullptr,*list=nullptr,*count=nullptr,*accepted=nullptr,*values=nullptr,*generic_flags=nullptr,*coupled_flags=nullptr,*float_list=nullptr,*float_count=nullptr;general::Work* works=nullptr;
    void release() noexcept {
        reduced.reset();
        cudaFree(cached_models);cudaFree(cache_class);
        for(float* p:{x32,rgb32,a32,bg32,out32,work32})cudaFree(p);
        cudaFree(work64);
        for(int* p:{raw,statuses,flags,list,count,accepted,values,generic_flags,coupled_flags,float_list,float_count})cudaFree(p);
        cudaFree(works);
    }
public:
    Workspace(int pixels,int layers,int constraints,bool use_fast,int blocks,general::Options opt,size_t budget=size_t(128)*1024*1024,int tile=0,const Model* host_models=nullptr):n(pixels),L(layers),C(constraints),state_tile(tile),fast(use_fast),options(opt),graph(host_models,layers),general32(n,L,C,opt,budget){
        if(tile!=0&&tile!=32&&tile!=128)throw std::invalid_argument("Invalid GPU state tile");
        cudaDeviceProp prop;int device;ck(cudaGetDevice(&device));ck(cudaGetDeviceProperties(&prop,device));inverse=fast&&!graph.gray&&custom_refinement::inverse_words(L)<=budget/8;words32=custom_refinement::words(L,inverse);general::Problem p;p.layers=L;words64=std::max(fast?words32:size_t(0),size_t(general::Optimizer<double,Prior,Model>::workspace(p,C,opt)));
        auto slots=[&](size_t words,size_t element){size_t max=budget/(words*element);if(!max)throw std::length_error("Custom workspace cannot hold one pixel");int capacity=int(std::min({size_t(n),max,size_t(prop.multiProcessorCount)*blocks*128}));if(capacity>=32)capacity=capacity/32*32;return capacity;};slots32=slots(words32,4);slots64=slots(words64,8);bytes32=fast?words32*slots32*4:0;bytes64=words64*slots64*8;
        if(fast){cube_slots=int(std::min({size_t(n),size_t(prop.multiProcessorCount)*blocks*128,bytes32/(reachable_cube::words(L)*4)}));if(cube_slots>=32)cube_slots=cube_slots/32*32;}
        try {
            ck(cudaMalloc(&x32,size_t(4)*L*n*4));ck(cudaMalloc(&out32,size_t(4)*L*n*4));ck(cudaMalloc(&rgb32,size_t(3)*n*4));ck(cudaMalloc(&a32,size_t(L)*n*4));ck(cudaMalloc(&bg32,size_t(3)*n*4));if(fast){ck(cudaMalloc(&work32,bytes32));ck(cudaMemset(work32,0,bytes32));}ck(cudaMalloc(&work64,bytes64));ck(cudaMemset(work64,0,bytes64));
            for(int** p:{&raw,&statuses,&flags,&list,&accepted,&values,&generic_flags,&coupled_flags,&float_list}){ck(cudaMalloc(p,size_t(n)*4));ck(cudaMemset(*p,0,size_t(n)*4));}ck(cudaMalloc(&count,4));ck(cudaMalloc(&float_count,4));ck(cudaMalloc(&works,size_t(n)*sizeof(general::Work)));ck(cudaMemset(works,0,size_t(n)*sizeof(general::Work)));ck(cudaMemset(out32,0,size_t(4)*L*n*4));
            if(fast&&graph.known&&graph.cube&&graph.source_over&&!graph.gray&&graph.gaussian&&register_first_gpu::size_supported(L)){ck(cudaMalloc(&cached_models,size_t(L)*sizeof(*cached_models)));ck(cudaMalloc(&cache_class,4));ck(cudaMemset(cache_class,0,4));}
            if(fast&&reduced_gpu::Plan::supports(host_models,L))reduced=std::make_unique<reduced_gpu::Plan>(n,host_models[1].mode,work32,bytes32);
            // Constructor memsets run on the default stream. A caller may submit
            // its first solve on a nonblocking stream immediately on return.
            ck(cudaStreamSynchronize(nullptr));
        } catch (...) {
            release();
            throw;
        }
    }
    ~Workspace(){release();}
    Workspace(const Workspace&)=delete;Workspace& operator=(const Workspace&)=delete;
    void solve(const Model* model,general::Problem p,const double* initial,const double* rgb,const double* alpha,const double* bg,double* output,int rounds,const int* input_status=nullptr,cudaStream_t stream=0){
        if(p.layers!=L)throw std::invalid_argument("Custom workspace layer mismatch");auto convert=[&](size_t size,const double* in,float* out){cast<<<std::min(size_t(65535),(size+255)/256),256,0,stream>>>(size,in,out);};convert(size_t(4)*L*n,initial,x32);convert(size_t(3)*n,rgb,rgb32);convert(size_t(L)*n,alpha,a32);convert(size_t(3)*n,bg,bg32);
        ck(cudaMemsetAsync(count,0,4,stream));ck(cudaMemsetAsync(generic_flags,0,size_t(n)*4,stream));ck(cudaMemsetAsync(works,0,size_t(n)*sizeof(general::Work),stream));
        ck(cudaMemsetAsync(coupled_flags,0,size_t(n)*4,stream));
        const int path=fast?float_path(p):0;
        const int* classification=(path==5||path==7)?cache_class:nullptr;
        if(path==3){reduced->run(model,p,x32,rgb32,out32,raw,accepted,values,coupled_flags,rounds,float(options.outer_tolerance),stream);}
        else if(fast){
            ck(cudaMemsetAsync(float_count,0,4,stream));
            if(path>=4){
                float tolerance=float(options.outer_tolerance);
                if(classification)register_first_gpu::configure<<<1,32,0,stream>>>(model,cached_models,cache_class,L);
                if(p.refinement){
                    if(classification)register_first_gpu::launch<true>(cached_models,classification,p,n,x32,rgb32,a32,bg32,out32,raw,accepted,values,tolerance,rounds,coupled_flags,stream);
                    reachable_cube::solve_kernel<float,true><<<(cube_slots+127)/128,128,0,stream>>>(model,p,n,cube_slots,x32,rgb32,a32,bg32,out32,work32,raw,accepted,values,tolerance,rounds,coupled_flags,state_tile,classification);
                }else{
                    if(classification)register_first_gpu::launch<false>(cached_models,classification,p,n,x32,rgb32,nullptr,bg32,out32,raw,accepted,values,tolerance,rounds,coupled_flags,stream);
                    reachable_cube::solve_kernel<float,false><<<(cube_slots+127)/128,128,0,stream>>>(model,p,n,cube_slots,x32,rgb32,nullptr,bg32,out32,work32,raw,accepted,values,tolerance,rounds,coupled_flags,state_tile,classification);
                }
            }
            else if(path==0)custom_refinement::kernel<<<(slots32+127)/128,128,0,stream>>>(model,p,n,slots32,x32,rgb32,a32,bg32,out32,work32,raw,accepted,values,float(options.outer_tolerance),rounds,coupled_flags,inverse,state_tile);
            else{
                if(path==1)custom_refinement::split_kernel<float,1><<<(slots32+127)/128,128,0,stream>>>(model,p,n,slots32,x32,rgb32,a32,bg32,out32,work32,raw,accepted,values,float(options.outer_tolerance),rounds,coupled_flags,inverse,state_tile,float_list,float_count,false);
                custom_refinement::split_kernel<float,2><<<(slots32+127)/128,128,0,stream>>>(model,p,n,slots32,x32,rgb32,a32,bg32,out32,work32,raw,accepted,values,float(options.outer_tolerance),rounds,coupled_flags,inverse,state_tile,float_list,float_count,path==1);
            }
        }
        else general32.solve(model,p,rgb32,a32,bg32,x32,out32,input_status,stream);
        guard<<<(n+127)/128,128,0,stream>>>(model,p,n,out32,rgb,alpha,bg,output,fast?raw:nullptr,statuses,flags,list,count,options.outer_tolerance,input_status,classification);
        repair<<<(slots64+127)/128,128,0,stream>>>(model,p,n,slots64,initial,rgb,alpha,bg,output,work64,words64,state_tile,list,count,statuses,generic_flags,coupled_flags,works,options.outer_tolerance,rounds,fast,inverse,options);ck(cudaGetLastError());
    }
    const int* status()const{return statuses;}const int* repair_flags()const{return flags;}const int* generic_repair_flags()const{return generic_flags;}const int* accepted_steps()const{return accepted;}const int* evaluations()const{return values;}
    const int* coupled_path_flags()const{return coupled_flags;}
    bool uses_inverse()const{return inverse;}
    int float_path(general::Problem p)const{return reduced&&reduced->eligible(p)?3:graph.path(p,inverse);}
    int reduced_capacity()const{return reduced?reduced->slots():0;}
    int cube_capacity()const{return cube_slots;}
    const int* register_configuration_class()const{return cache_class;}
    int state_tile_size()const{return state_tile;}
    size_t allocated_bytes()const{return bytes32+bytes64+general32.allocated_bytes()+(reduced?reduced->extra_bytes():0)+(cached_models?size_t(L)*sizeof(*cached_models)+4:0);}int float_slots()const{return slots32;}int double_slots()const{return slots64;}
};
}
