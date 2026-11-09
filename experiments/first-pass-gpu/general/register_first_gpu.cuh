#pragma once
#include "register_first.cuh"
namespace register_first_gpu {
inline bool size_supported(int layers){return layers==2||layers==3||layers==5||layers==7;}

// Refresh numeric parameters on the GPU on EVERY solve/replay. Classification:
// 1 = Gaussian register kernel, 0 = full-prior cube fallback, -1 = the device
// topology disagrees with the immutable host plan (caller must rebuild it).
__global__ void configure(const custom_backend::Model* input,register_first::Model<float>* output,int* classification,int layers){
    if(threadIdx.x==0)*classification=1;__syncthreads();int l=threadIdx.x;
    if(l<layers){const auto& m=input[l];output[l]=register_first::cache<float>(m);int kind=m.term_count==0?1:0;
        if(m.gray||(l&&(m.X!=1||m.Y!=1||m.Z!=1||m.mode<0||m.mode>11||m.mode==6||m.mode==7)))kind=-1;
        if(kind!=1)atomicMin(classification,kind);
    }
}
template<int L,bool Refine>__global__ __launch_bounds__(128,4) void solve_kernel(
    const register_first::Model<float>* model,const int* classification,general::Problem p,int n,
    const float* initial,const float* target,const float* alpha,const float* bg,float* output,int* status,
    int* accepted,int* values,float tolerance,int rounds,int* paths){
    if(*classification!=1)return;
    __shared__ register_first::Model<float> shared[L];if(threadIdx.x<L)shared[threadIdx.x]=model[threadIdx.x];__syncthreads();int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    register_first::Point<float,L> point;float wanted[3],background[3];
    register_first::each<0,L>([&](auto l) __attribute__((always_inline)){point.a[l]=(Refine?alpha:initial)[size_t(l)*n+i];register_first::each<0,3>([&](auto r) __attribute__((always_inline)){point.c[l][r]=initial[size_t(L+3*l+r)*n+i];});});
    register_first::each<0,3>([&](auto r) __attribute__((always_inline)){wanted[r]=target[size_t(r)*n+i];background[r]=bg[size_t(r)*n+i];});
    int steps,evaluations,code;
    if constexpr(Refine)code=register_first::improve(shared,p,point,wanted,background,tolerance,rounds,steps,evaluations);
    else code=register_first::first(shared,p,point,wanted,background,tolerance,rounds,steps,evaluations);
    status[i]=code;accepted[i]=steps;values[i]=evaluations;
    register_first::each<0,L>([&](auto l) __attribute__((always_inline)){output[size_t(l)*n+i]=point.a[l];register_first::each<0,3>([&](auto r) __attribute__((always_inline)){output[size_t(L+3*l+r)*n+i]=point.c[l][r];});});if(paths)paths[i]=0;
}
template<bool Refine>inline void launch(const register_first::Model<float>* model,const int* classification,general::Problem p,int n,
    const float* initial,const float* target,const float* alpha,const float* bg,float* output,int* status,int* accepted,int* values,float tolerance,int rounds,int* paths,cudaStream_t stream){
    switch(p.layers){
        case 2:solve_kernel<2,Refine><<<(n+127)/128,128,0,stream>>>(model,classification,p,n,initial,target,alpha,bg,output,status,accepted,values,tolerance,rounds,paths);break;
        case 3:solve_kernel<3,Refine><<<(n+127)/128,128,0,stream>>>(model,classification,p,n,initial,target,alpha,bg,output,status,accepted,values,tolerance,rounds,paths);break;
        case 5:solve_kernel<5,Refine><<<(n+127)/128,128,0,stream>>>(model,classification,p,n,initial,target,alpha,bg,output,status,accepted,values,tolerance,rounds,paths);break;
        case 7:solve_kernel<7,Refine><<<(n+127)/128,128,0,stream>>>(model,classification,p,n,initial,target,alpha,bg,output,status,accepted,values,tolerance,rounds,paths);break;
        default:throw std::invalid_argument("Unsupported register solver size; use the runtime fallback");
    }
}
template<bool Refine>inline void resources(int layers,cudaFuncAttributes& attributes,int& active){
    switch(layers){
        case 2:ck(cudaFuncGetAttributes(&attributes,solve_kernel<2,Refine>));ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,solve_kernel<2,Refine>,128,0));break;
        case 3:ck(cudaFuncGetAttributes(&attributes,solve_kernel<3,Refine>));ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,solve_kernel<3,Refine>,128,0));break;
        case 5:ck(cudaFuncGetAttributes(&attributes,solve_kernel<5,Refine>));ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,solve_kernel<5,Refine>,128,0));break;
        case 7:ck(cudaFuncGetAttributes(&attributes,solve_kernel<7,Refine>));ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,solve_kernel<7,Refine>,128,0));break;
        default:throw std::invalid_argument("Invalid register solver resource size");
    }
}
}
