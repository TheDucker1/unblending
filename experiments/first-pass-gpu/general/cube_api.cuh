#pragma once
#include "custom_refinement.cuh"
namespace reachable_cube {
// Opaque source-over with free RGB background and cube-preserving modes.
// Fixed-alpha refinement uses the same reachability proof.
template<class T,bool Refine>__global__ __launch_bounds__(128,4) void solve_kernel(
    const custom_backend::Model* model,general::Problem p,int n,int slots,
    const T* initial,const T* target,const T* alpha,const T* bg,T* output,T* memory,
    int* status,int* accepted,int* values,T tolerance,int rounds,int* paths,int tile,const int* classification=nullptr){
    int kind=classification?*classification:0;if(kind==1)return;
    int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=slots)return;
    if(kind<0){for(int i=slot;i<n;i+=slots){for(int k=0;k<4*p.layers;++k)output[size_t(k)*n+i]=initial[size_t(k)*n+i];status[i]=8;accepted[i]=values[i]=0;if(paths)paths[i]=0;}return;}
    auto work=custom_refinement::state_view(memory,slot,slots,words(p.layers),tile);
    for(int i=slot;i<n;i+=slots){auto cv=[&](const T* x){return general::View<const T>{x+i,size_t(n)};};
        general::View<T> point{output+i,size_t(n)};
        int steps,evaluations;
        if constexpr(Refine){
            status[i]=custom_refinement::improve<T,true>(model,p,cv(initial),cv(target),cv(alpha),cv(bg),point,work,tolerance,rounds,steps,evaluations);
        }else{
            status[i]=custom_refinement::first<T,true>(model,p,cv(initial),cv(target),cv(bg),point,work,tolerance,rounds,steps,evaluations);
        }
        accepted[i]=steps;values[i]=evaluations;if(paths)paths[i]=0;
    }
}
}
