#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace probe {
enum Event {Norm,RestoreNorm,RestoreCall,RestoreSeed,RestoreIteration,RestoreTrial,
    GradientFallback,OuterIteration,OuterTrial,OuterRestore,InitialRestore,Energy,
    Metric,ActivePass,Accepted,RestoreStalled,NormWarpEquivalent,RestoreIterationLimit,Events};
struct Trace {unsigned long long c[Events]{};int depth=0;};
static_assert(sizeof(Trace)==152,"trace binary layout");
#ifdef PROBE_TRACE
__device__ Trace* traces;
thread_local Trace* host_trace=nullptr;
__host__ __device__ inline void hit(Event e){
#ifdef __CUDA_ARCH__
    Trace* t=traces+blockIdx.x*blockDim.x+threadIdx.x;
#else
    Trace* t=host_trace;if(!t)return;
#endif
    ++t->c[e];
    if(e==RestoreCall)++t->depth;
    if(e==Norm){if(t->depth)++t->c[RestoreNorm];
#ifdef __CUDA_ARCH__
        t->c[NormWarpEquivalent]+=1048576u/__popc(__activemask());
#endif
    }
}
__host__ __device__ inline void end_restore(){
#ifdef __CUDA_ARCH__
    --traces[blockIdx.x*blockDim.x+threadIdx.x].depth;
#else
    if(host_trace)--host_trace->depth;
#endif
}
#else
__host__ __device__ inline void hit(Event){}
__host__ __device__ inline void end_restore(){}
#endif
}
