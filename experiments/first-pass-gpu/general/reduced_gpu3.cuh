#pragma once
#include "register_direction3.cuh"
#include <stdexcept>
namespace reduced_gpu {
using Model=custom_backend::Model;using Point=register_three::Point;using V=register_three::V;
// Full-image planes at the boundary; bounded, independent scratch slots.
struct Frame {
 int n,begin,size,capacity;const float *initial,*target;float* point;int *status,*accepted,*values,*coupled;
 float *delta,*anchor,*energy,*limits,*candidate,*candidate_energy;unsigned char* called;int *ready,*trials,*counts;
};
__device__ inline void push(int* queue,int* count,int pixel){unsigned mask=__activemask();int lane=threadIdx.x&31,leader=__ffs(mask)-1,base=0;if(lane==leader)base=atomicAdd(count,__popc(mask));base=__shfl_sync(mask,base,leader);queue[base+__popc(mask&((1u<<lane)-1))]=pixel;}
__device__ inline Point load7(const float* v,int stride,int i){return {v[i],v[size_t(stride)+i],v[size_t(2)*stride+i],{v[size_t(3)*stride+i],v[size_t(4)*stride+i],v[size_t(5)*stride+i]},v[size_t(6)*stride+i]};}
__device__ inline void store7(float* v,int stride,int i,Point x){v[i]=x.a;v[size_t(stride)+i]=x.u;v[size_t(2)*stride+i]=x.bg;v[size_t(3)*stride+i]=x.c.r;v[size_t(4)*stride+i]=x.c.g;v[size_t(5)*stride+i]=x.c.b;v[size_t(6)*stride+i]=x.top;}
__device__ inline V target(Frame f,int i){return {f.target[i],f.target[size_t(f.n)+i],f.target[size_t(2)*f.n+i]};}
__device__ inline Point point(Frame f,int i){return register_three::load(general::View<const float>{f.point+i,size_t(f.n)});}
__device__ inline void put(Frame f,int i,Point x){register_three::store(general::View<float>{f.point+i,size_t(f.n)},x);}
template<int Mode>__global__ __launch_bounds__(128,4) void initialize(const Model* model,general::Problem p,Frame f,float tolerance){
 int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=f.size)return;int i=f.begin+slot;Point start=register_three::initial(general::View<const float>{f.initial+i,size_t(f.n)});float norm;Point x=register_three::restore<Mode>(model,p,start,target(f,i),64,norm,tolerance);put(f,i,x);f.status[i]=3;f.accepted[i]=f.values[i]=0;f.coupled[i]=1;
 if(norm<=tolerance*tolerance){f.status[i]=0;push(f.ready,f.counts,i);}
}
__global__ void initial_energy(const Model* model,general::Problem p,Frame f){int job=blockIdx.x*blockDim.x+threadIdx.x;if(job>=*f.counts)return;int i=f.ready[job],slot=i-f.begin;float value=register_three::energy(model,p,point(f,i));f.energy[slot]=value;f.values[i]=1;if(!std::isfinite(value))f.status[i]=2;}
template<int Mode>__global__ __launch_bounds__(128,4) void direction(const Model* model,general::Problem p,Frame f){
 int job=blockIdx.x*blockDim.x+threadIdx.x;if(job>=*f.counts)return;int i=f.ready[job],slot=i-f.begin;if(f.status[i]!=0)return;Point x=point(f,i),d;V anchor;
 if(!register_three::proposal(model,p,x,target(f,i),d,anchor,Mode,Mode))return;float limit;if(!register_three::limit_step(x,d,limit))return;store7(f.delta,f.capacity,slot,d);f.anchor[slot]=anchor.r;f.anchor[size_t(f.capacity)+slot]=anchor.g;f.anchor[size_t(2)*f.capacity+slot]=anchor.b;f.limits[slot]=limit;push(f.trials,f.counts+1,i);
}
template<int Mode>__global__ __launch_bounds__(128,4) void trial(const Model* model,general::Problem p,Frame f,float tolerance){
 int job=blockIdx.x*blockDim.x+threadIdx.x,count=f.counts[1];if(job>=6*count)return;int back=job/count,i=f.trials[job-back*count],slot=i-f.begin;
 float factor=__uint_as_float(0x3f800000u-(unsigned(back)<<23));Point x=register_three::advance(point(f,i),load7(f.delta,f.capacity,slot),f.limits[slot]*factor);V anchor{f.anchor[slot],f.anchor[size_t(f.capacity)+slot],f.anchor[size_t(2)*f.capacity+slot]};
 float norm=register_three::trial_restore(x,target(f,i),anchor,Mode,Mode,tolerance);bool called=norm<=tolerance*tolerance;float value=called?register_three::energy(model,p,x):INFINITY;size_t offset=size_t(back)*f.capacity+slot;
 f.called[offset]=called;f.candidate_energy[offset]=std::isfinite(value)?value:INFINITY;if(std::isfinite(value))store7(f.candidate+size_t(back)*7*f.capacity,f.capacity,slot,x);
}
__global__ void commit(Frame f){int job=blockIdx.x*blockDim.x+threadIdx.x;if(job>=f.counts[1])return;int i=f.trials[job],slot=i-f.begin,calls=0;for(int back=0;back<6;++back)calls+=f.called[size_t(back)*f.capacity+slot];f.values[i]+=calls;
 for(int back=0;back<6;++back){float value=f.candidate_energy[size_t(back)*f.capacity+slot];if(value<f.energy[slot]){put(f,i,load7(f.candidate+size_t(back)*7*f.capacity,f.capacity,slot));f.energy[slot]=value;++f.accepted[i];push(f.ready,f.counts,i);return;}}
}
class Plan {
 int n,capacity,mode;float* scratch;int *ready=nullptr,*trials=nullptr,*counts=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t replay=nullptr;cudaStream_t capture_stream=nullptr;
 const Model* cached_model=nullptr;general::Problem cached_problem{};int cached_rounds=-1;float cached_tolerance=0;Frame cached_frame{};
 static bool same(general::Problem a,general::Problem b){return a.layers==b.layers&&a.refinement==b.refinement&&a.opaque==b.opaque&&a.smooth==b.smooth&&a.fixed_alpha==b.fixed_alpha&&a.sparsity==b.sparsity&&a.minimum==b.minimum&&a.sparsity_weight==b.sparsity_weight;}
 void clear(){if(replay){ck(cudaGraphExecDestroy(replay));replay=nullptr;}if(graph){ck(cudaGraphDestroy(graph));graph=nullptr;}}
 Frame frame(const float* initial,const float* rgb,float* output,int* status,int* accepted,int* values,int* coupled)const{
  Frame f{};f.n=n;f.capacity=capacity;f.initial=initial;f.target=rgb;f.point=output;f.status=status;f.accepted=accepted;f.values=values;f.coupled=coupled;f.delta=scratch;f.anchor=scratch+size_t(7)*capacity;f.energy=scratch+size_t(10)*capacity;f.limits=scratch+size_t(11)*capacity;f.candidate=scratch+size_t(12)*capacity;f.candidate_energy=scratch+size_t(54)*capacity;f.called=reinterpret_cast<unsigned char*>(scratch+size_t(60)*capacity);f.ready=ready;f.trials=trials;f.counts=counts;return f;
 }
 void enqueue(const Model* model,general::Problem p,Frame f,int rounds,float tolerance,cudaStream_t stream){
  for(int begin=0;begin<n;begin+=capacity){f.begin=begin;f.size=std::min(capacity,n-begin);int blocks=(f.size+127)/128;ck(cudaMemsetAsync(counts,0,2*sizeof(int),stream));
   if(mode==6)initialize<6><<<blocks,128,0,stream>>>(model,p,f,tolerance);else initialize<7><<<blocks,128,0,stream>>>(model,p,f,tolerance);initial_energy<<<blocks,128,0,stream>>>(model,p,f);
   for(int r=0;r<rounds;++r){ck(cudaMemsetAsync(counts+1,0,sizeof(int),stream));if(mode==6)direction<6><<<blocks,128,0,stream>>>(model,p,f);else direction<7><<<blocks,128,0,stream>>>(model,p,f);ck(cudaMemsetAsync(counts,0,sizeof(int),stream));
    if(mode==6)trial<6><<<(6*f.size+127)/128,128,0,stream>>>(model,p,f,tolerance);else trial<7><<<(6*f.size+127)/128,128,0,stream>>>(model,p,f,tolerance);commit<<<blocks,128,0,stream>>>(f);
   }
  }
  ck(cudaGetLastError());
 }
public:
 static bool supports(const Model* models,int layers){if(!models||layers!=3)return false;general::Problem p{3};return register_three::eligible(models,p);}
 Plan(int pixels,int selected_mode,float* reusable,size_t bytes):n(pixels),capacity(int(std::min(size_t(pixels),bytes/(60*sizeof(float)+6*sizeof(unsigned char))))),mode(selected_mode),scratch(reusable){
  if(capacity<1)throw std::length_error("Reduced GPU scratch cannot hold a pixel");
  try {
   ck(cudaMalloc(&ready,size_t(capacity)*sizeof(int)));
   ck(cudaMalloc(&trials,size_t(capacity)*sizeof(int)));
   ck(cudaMalloc(&counts,2*sizeof(int)));
   ck(cudaStreamCreateWithFlags(&capture_stream,cudaStreamNonBlocking));
  } catch (...) {
   cudaFree(ready);cudaFree(trials);cudaFree(counts);
   throw;
  }
 }
 ~Plan(){if(replay)cudaGraphExecDestroy(replay);if(graph)cudaGraphDestroy(graph);if(capture_stream)cudaStreamDestroy(capture_stream);cudaFree(ready);cudaFree(trials);cudaFree(counts);}Plan(const Plan&)=delete;Plan& operator=(const Plan&)=delete;
 bool eligible(general::Problem p)const{return p.layers==3&&p.opaque&&!p.refinement;}int slots()const{return capacity;}int blend_mode()const{return mode;}size_t extra_bytes()const{return size_t(2)*capacity*sizeof(int)+2*sizeof(int);}
 void run(const Model* model,general::Problem p,const float* initial,const float* rgb,float* output,int* status,int* accepted,int* values,int* coupled,int rounds,float tolerance,cudaStream_t stream){
  Frame f=frame(initial,rgb,output,status,accepted,values,coupled);cudaStreamCaptureStatus capturing;ck(cudaStreamIsCapturing(stream,&capturing));if(capturing!=cudaStreamCaptureStatusNone){enqueue(model,p,f,rounds,tolerance,stream);return;}
  bool hit=replay&&cached_model==model&&same(cached_problem,p)&&cached_rounds==rounds&&cached_tolerance==tolerance&&cached_frame.initial==initial&&cached_frame.target==rgb&&cached_frame.point==output&&cached_frame.status==status&&cached_frame.accepted==accepted&&cached_frame.values==values&&cached_frame.coupled==coupled;
  if(!hit){clear();ck(cudaStreamBeginCapture(capture_stream,cudaStreamCaptureModeThreadLocal));enqueue(model,p,f,rounds,tolerance,capture_stream);ck(cudaStreamEndCapture(capture_stream,&graph));ck(cudaGraphInstantiate(&replay,graph,nullptr,nullptr,0));cached_model=model;cached_problem=p;cached_rounds=rounds;cached_tolerance=tolerance;cached_frame=f;}
  ck(cudaGraphLaunch(replay,stream));
 }
};
}
