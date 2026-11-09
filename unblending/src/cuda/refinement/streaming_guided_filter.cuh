#pragma once
#include "filtered_matte_targets.cuh"

namespace general {
// Four scalar channels per batch bound scratch memory independently of layers.
// Statistics stay double; float AND double images/layers remain native to the
// caller. In particular a double pipeline never roundtrips through FP32 here.
template<class T>HD double guidance_value(const T* rgb,int pixel,int plane,int n){
    if(plane<3)return double(rgb[size_t(plane)*n+pixel]);
    const int r[6]={0,0,0,1,1,2},s[6]={0,1,2,1,2,2};
    return double(rgb[size_t(r[plane-3])*n+pixel])*double(rgb[size_t(s[plane-3])*n+pixel]);
}
template<class T>HD double channel_value(const T* first,int pixel,int channel,int n){return double(first[size_t(channel)*n+pixel]);}
template<class T>HD double reconstructed_channel(const T* rgb,const T* first,const double* mean,int pixel,int channel,int local,int n,bool identity){
    if(identity)return configured::clip(channel_value(first,pixel,channel,n));
    int base=4*local;double result=mean[size_t(base)*n+pixel];
    for(int r=0;r<3;++r)result+=mean[size_t(base+r+1)*n+pixel]*double(rgb[size_t(r)*n+pixel]);return configured::clip(result);
}
template<class T>__global__ void stream_guidance(const T* rgb,double* raw,int n){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<9*n)raw[j]=guidance_value(rgb,j%n,j/n,n);}
template<class T>__global__ void stream_pack(const T* rgb,const T* first,double* raw,int n,int offset,int count){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=4*count*n)return;int plane=j/n,pixel=j%n,k=plane%4,channel=offset+plane/4;double v=channel_value(first,pixel,channel,n);
    raw[j]=k?v*double(rgb[size_t(k-1)*n+pixel]):v;
}
template<class T>__global__ void stream_evaluate(const T* rgb,const T* first,const double* mean,T* targets,double* sums,int* status,int n,int L,int offset,int count,bool plus,bool identity){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;double sum=0;
    for(int local=0;local<count;++local){int channel=offset+local;double value=reconstructed_channel(rgb,first,mean,p,channel,local,n,identity);targets[size_t(channel)*n+p]=T(value);
        if(channel<L)sum+=value;if(channel==0)status[p]=!plus&&std::abs(value-1.)>=1e-5?int(NormalizationStatus::NonopaqueSourceOver):0;}
    if(plus)sums[p]+=sum;
}
template<class T>__global__ void stream_finish(const T* first,T* targets,const double* sums,int* status,int n,int L,bool plus,bool smooth){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
    if(plus){double sum=sums[p];if(sum>0)for(int l=0;l<L;++l)targets[size_t(l)*n+p]=T(double(targets[size_t(l)*n+p])/sum);else status[p]=int(NormalizationStatus::ZeroPlusSum);}
    if(!smooth)for(int r=0;r<3;++r)targets[size_t(L+r)*n+p]=first[size_t(L+r)*n+p];
}

template<class T>class StreamingFilterGPU {
    int n,L,channels;bool plus,smooth;
    twostage::FilterGPU box;
    double* sums=nullptr;int* status=nullptr;
public:
    template<class Model>StreamingFilterGPU(int w,int h,int layers,bool opaque,bool bg,const Model* model):n(w*h),L(layers),channels(layers+(bg?3:0)),plus(normalization_kind(model,layers,opaque,bg)),smooth(bg),box(w,h,5,false){
        try {
            twostage::cuda_check(cudaMalloc(&sums,size_t(n)*sizeof(double)));
            twostage::cuda_check(cudaMalloc(&status,size_t(n)*sizeof(int)));
        } catch (...) {
            cudaFree(sums);
            cudaFree(status);
            throw;
        }
    }
    ~StreamingFilterGPU(){cudaFree(sums);cudaFree(status);}
    const int* device_status()const{return status;}
    void run(const T* rgb,const T* first,T* targets,int radius,cudaStream_t stream=0){
        auto raw=box.raw_buffer(),mean=box.mean_buffer(),info=box.guidance_buffer();
        twostage::cuda_check(cudaMemsetAsync(sums,0,size_t(n)*sizeof(double),stream));
        if(radius){stream_guidance<<<(9*n+255)/256,256,0,stream>>>(rgb,raw,n);box.box(raw,mean,9,radius,stream);twostage::invert_guidance<<<(n+255)/256,256,0,stream>>>(mean,info,n);}
        for(int offset=0;offset<channels;offset+=4){int count=channels-offset<4?channels-offset:4;
            if(radius){stream_pack<<<(4*count*n+255)/256,256,0,stream>>>(rgb,first,raw,n,offset,count);box.box(raw,mean,4*count,radius,stream);
                twostage::make_coefficients<<<(count*n+255)/256,256,0,stream>>>(mean,info,raw,n,count);box.box(raw,mean,4*count,radius,stream);}
            stream_evaluate<<<(n+255)/256,256,0,stream>>>(rgb,first,mean,targets,sums,status,n,L,offset,count,plus,radius==0);
        }
        stream_finish<<<(n+255)/256,256,0,stream>>>(first,targets,sums,status,n,L,plus,smooth);twostage::cuda_check(cudaGetLastError());
    }
};

template<class T>class StreamingFilterCPU {
    int n,L,channels;bool plus,smooth;twostage::Pool& pool;twostage::FilterCPU box;
    std::vector<double> sums;std::vector<int> status;
public:
    template<class Model>StreamingFilterCPU(int w,int h,int layers,bool opaque,bool bg,const Model* model,twostage::Pool& workers):n(w*h),L(layers),channels(layers+(bg?3:0)),plus(normalization_kind(model,layers,opaque,bg)),smooth(bg),pool(workers),box(w,h,5,false,workers),sums(n),status(n){}
    const std::vector<int>& statuses()const{return status;}
    void run(const T* rgb,const T* first,T* targets,int radius){
        auto raw=box.raw_buffer(),mean=box.mean_buffer(),info=box.guidance_buffer();std::fill(sums.begin(),sums.end(),0.);
        if(radius){pool.run(9*n,256,[&](int b,int e){for(int j=b;j<e;++j)raw[j]=guidance_value(rgb,j%n,j/n,n);});box.box(raw,mean,9,radius);
            pool.run(n,256,[&](int b,int e){for(int p=b;p<e;++p)twostage::guidance_inverse(p,n,mean,info);});}
        for(int offset=0;offset<channels;offset+=4){int count=channels-offset<4?channels-offset:4;
            if(radius){pool.run(4*count*n,256,[&](int b,int e){for(int j=b;j<e;++j){int plane=j/n,p=j%n,k=plane%4;double v=channel_value(first,p,offset+plane/4,n);raw[j]=k?v*double(rgb[size_t(k-1)*n+p]):v;}});box.box(raw,mean,4*count,radius);
                pool.run(count*n,256,[&](int b,int e){for(int j=b;j<e;++j)twostage::coefficients(j%n,j/n,n,mean,info,raw);});box.box(raw,mean,4*count,radius);}
            pool.run(n,256,[&](int b,int e){for(int p=b;p<e;++p){double sum=0;for(int local=0;local<count;++local){int channel=offset+local;double value=reconstructed_channel(rgb,first,mean,p,channel,local,n,radius==0);targets[size_t(channel)*n+p]=T(value);
                    if(channel<L)sum+=value;if(channel==0)status[p]=!plus&&std::abs(value-1.)>=1e-5?int(NormalizationStatus::NonopaqueSourceOver):0;}
                if(plus)sums[p]+=sum;}});
        }
        pool.run(n,256,[&](int b,int e){for(int p=b;p<e;++p){if(plus){double sum=sums[p];if(sum>0)for(int l=0;l<L;++l)targets[size_t(l)*n+p]=T(double(targets[size_t(l)*n+p])/sum);else status[p]=int(NormalizationStatus::ZeroPlusSum);}
            if(!smooth)for(int r=0;r<3;++r)targets[size_t(L+r)*n+p]=first[size_t(L+r)*n+p];}});
    }
};
}
