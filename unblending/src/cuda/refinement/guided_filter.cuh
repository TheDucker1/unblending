#pragma once
#include "filter_worker_pool.hpp"
#include "compositing/fixed_stack.cuh"
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace twostage {
inline void cuda_check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
HD inline int bounded(int v,int n){return v<0?0:(v>=n?n-1:v);}
// Double-precision statistics protect the small regularized covariance from
// cancellation. Solver inputs/outputs remain FP32. Shared on CPU and device.
HD inline void guidance_product(int p,int plane,int n,const float* rgb,double& value){
    if(plane<3){value=rgb[plane*n+p];return;}
    const int r[6]={0,0,0,1,1,2},s[6]={0,1,2,1,2,2};
    value=double(rgb[r[plane-3]*n+p])*double(rgb[s[plane-3]*n+p]);
}
HD inline void guidance_inverse(int p,int n,const double* mean,double* info){
    double r=mean[p],g=mean[n+p],b=mean[2*n+p];
    double a=mean[3*n+p]-r*r+1e-4,d=mean[4*n+p]-r*g,e=mean[5*n+p]-r*b;
    double f=mean[6*n+p]-g*g+1e-4,h=mean[7*n+p]-g*b,i=mean[8*n+p]-b*b+1e-4;
    double A=f*i-h*h,B=e*h-d*i,C=d*h-e*f,D=a*i-e*e,E=d*e-a*h,F=a*f-d*d;
    double determinant=a*A+d*B+e*C;
    info[p]=r;info[n+p]=g;info[2*n+p]=b;
    info[3*n+p]=A/determinant;info[4*n+p]=B/determinant;info[5*n+p]=C/determinant;
    info[6*n+p]=D/determinant;info[7*n+p]=E/determinant;info[8*n+p]=F/determinant;
}
HD inline double source_channel(int p,int ch,int n,int layers,const float* first){
    return ch<layers-1?first[(ch+1)*n+p]:first[(layers+ch-(layers-1))*n+p];
}
HD inline void coefficients(int p,int channel,int n,const double* mean,const double* info,double* out){
    int base=4*channel*n;double mp=mean[base+p];
    double x=mean[base+n+p]-info[p]*mp,y=mean[base+2*n+p]-info[n+p]*mp,z=mean[base+3*n+p]-info[2*n+p]*mp;
    double a=info[3*n+p]*x+info[4*n+p]*y+info[5*n+p]*z;
    double b=info[4*n+p]*x+info[6*n+p]*y+info[7*n+p]*z;
    double c=info[5*n+p]*x+info[7*n+p]*y+info[8*n+p]*z;
    out[base+p]=mp-a*info[p]-b*info[n+p]-c*info[2*n+p];
    out[base+n+p]=a;out[base+2*n+p]=b;out[base+3*n+p]=c;
}
HD inline float filtered_value(int p,int ch,int n,const float* rgb,const double* means){
    int base=4*ch*n;double q=means[base+p];
    for(int r=0;r<3;++r)q+=means[base+(r+1)*n+p]*double(rgb[r*n+p]);
    return float(configured::clip(q));
}

// Each row is scanned cooperatively. Transposition makes the vertical pass use
// coalesced rows too. Window sums include repeated edge samples exactly.
__global__ void scan_rows(const double* input,double* prefix,int width,int rows){
    using Scan=cub::BlockScan<double,256>;__shared__ typename Scan::TempStorage storage;
    int row=blockIdx.x;if(row>=rows)return;double carry=0.;
    for(int start=0;start<width;start+=256){int x=start+threadIdx.x;double v=x<width?input[size_t(row)*width+x]:0.,sum,aggregate;
        Scan(storage).InclusiveSum(v,sum,aggregate);
        if(x<width)prefix[size_t(row)*width+x]=carry+sum;
        carry+=aggregate;__syncthreads();
    }
}
__global__ void window_rows(const double* input,const double* prefix,double* out,int width,int count,int radius){
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count)return;
    int x=index%width,base=index-x,left=x-radius,right=x+radius;
    int lo=left<0?0:left,hi=right>=width?width-1:right;
    double value=prefix[base+hi]-(lo?prefix[base+lo-1]:0.);
    if(left<0)value+=double(-left)*input[base];
    if(right>=width)value+=double(right-width+1)*input[base+width-1];
    out[index]=value/double(2*radius+1);
}
__global__ void transpose_planes(const double* input,double* out,int width,int height,int planes){
    __shared__ double tile[32][33];int x=blockIdx.x*32+threadIdx.x,y=blockIdx.y*32+threadIdx.y,plane=blockIdx.z;
    for(int k=0;k<32;k+=8)if(x<width&&y+k<height)tile[threadIdx.y+k][threadIdx.x]=input[(size_t(plane)*height+y+k)*width+x];
    __syncthreads();x=blockIdx.y*32+threadIdx.x;y=blockIdx.x*32+threadIdx.y;
    for(int k=0;k<32;k+=8)if(x<height&&y+k<width)out[(size_t(plane)*width+y+k)*height+x]=tile[threadIdx.x][threadIdx.y+k];
}
__global__ void pack_guidance(const float* rgb,double* raw,int n){int j=blockIdx.x*256+threadIdx.x;if(j<9*n)guidance_product(j%n,j/n,n,rgb,raw[j]);}
__global__ void invert_guidance(const double* means,double* info,int n){int p=blockIdx.x*256+threadIdx.x;if(p<n)guidance_inverse(p,n,means,info);}
__global__ void pack_channels(const float* rgb,const float* first,double* raw,int n,int layers,int channels){
    int j=blockIdx.x*256+threadIdx.x;if(j>=4*channels*n)return;int plane=j/n,p=j%n,k=plane%4,ch=plane/4;
    double value=source_channel(p,ch,n,layers,first);raw[j]=k?value*double(rgb[(k-1)*n+p]):value;
}
__global__ void make_coefficients(const double* mean,const double* info,double* raw,int n,int channels){
    int j=blockIdx.x*256+threadIdx.x;if(j<channels*n)coefficients(j%n,j/n,n,mean,info,raw);
}
__global__ void evaluate_filter(const float* rgb,const float* first,const double* means,float* targets,int n,int layers,bool smooth,bool identity){
    int p=blockIdx.x*256+threadIdx.x;if(p>=n)return;targets[p]=1.;
    for(int l=1;l<layers;++l)targets[l*n+p]=identity?first[l*n+p]:filtered_value(p,l-1,n,rgb,means);
    for(int r=0;r<3;++r)targets[(layers+r)*n+p]=(smooth&&!identity)?filtered_value(p,layers-1+r,n,rgb,means):first[(layers+r)*n+p];
}
class FilterGPU {
    int width,height,n,channels,layers,capacity;bool smooth;
    double *raw=nullptr,*means=nullptr,*info=nullptr,*horizontal=nullptr,*transposed=nullptr,*prefix=nullptr;
    void release() noexcept {for(auto p:{raw,means,info,horizontal,transposed,prefix})cudaFree(p);}
public:
    FilterGPU(int w,int h,int L,bool bg):width(w),height(h),n(w*h),channels(L-1+(bg?3:0)),layers(L),capacity(std::max(9,4*channels)),smooth(bg){
        try {
            for(double** p:{&raw,&means,&horizontal,&transposed,&prefix})cuda_check(cudaMalloc(p,size_t(capacity)*n*sizeof(double)));
            cuda_check(cudaMalloc(&info,size_t(9)*n*sizeof(double)));
        } catch (...) {
            release();
            throw;
        }
    }
    ~FilterGPU(){release();}
    double* raw_buffer(){return raw;}
    double* mean_buffer(){return means;}
    double* guidance_buffer(){return info;}
    void box(const double* input,double* output,int planes,int radius,cudaStream_t stream=0){
        int count=planes*n;scan_rows<<<height*planes,256,0,stream>>>(input,prefix,width,height*planes);
        window_rows<<<(count+255)/256,256,0,stream>>>(input,prefix,horizontal,width,count,radius);
        transpose_planes<<<dim3((width+31)/32,(height+31)/32,planes),dim3(32,8),0,stream>>>(horizontal,transposed,width,height,planes);
        scan_rows<<<width*planes,256,0,stream>>>(transposed,prefix,height,width*planes);
        window_rows<<<(count+255)/256,256,0,stream>>>(transposed,prefix,horizontal,height,count,radius);
        transpose_planes<<<dim3((height+31)/32,(width+31)/32,planes),dim3(32,8),0,stream>>>(horizontal,output,height,width,planes);
        cuda_check(cudaGetLastError());
    }
    void run(const float* rgb,const float* first,float* targets,int radius,cudaStream_t stream=0){
        if(radius>0){
            pack_guidance<<<(9*n+255)/256,256,0,stream>>>(rgb,raw,n);box(raw,means,9,radius,stream);
            invert_guidance<<<(n+255)/256,256,0,stream>>>(means,info,n);
            pack_channels<<<(4*channels*n+255)/256,256,0,stream>>>(rgb,first,raw,n,layers,channels);box(raw,means,4*channels,radius,stream);
            make_coefficients<<<(channels*n+255)/256,256,0,stream>>>(means,info,raw,n,channels);box(raw,means,4*channels,radius,stream);
        }
        evaluate_filter<<<(n+255)/256,256,0,stream>>>(rgb,first,means,targets,n,layers,smooth,radius==0);cuda_check(cudaGetLastError());
    }
};

class FilterCPU {
    int width,height,n,channels,layers,capacity;bool smooth;Pool& pool;
    std::vector<double> raw,means,info,horizontal;
public:
    FilterCPU(int w,int h,int L,bool bg,Pool& workers):width(w),height(h),n(w*h),channels(L-1+(bg?3:0)),layers(L),capacity(std::max(9,4*channels)),smooth(bg),pool(workers),
        raw(size_t(capacity)*n),means(raw.size()),info(size_t(9)*n),horizontal(raw.size()){}
    double* raw_buffer(){return raw.data();}
    double* mean_buffer(){return means.data();}
    double* guidance_buffer(){return info.data();}
    void box(const double* input,double* output,int planes,int radius){
        // Sliding sums are O(pixels), including windows larger than the image.
        pool.run(planes*height,1,[&](int begin,int end){for(int row=begin;row<end;++row){
            size_t base=size_t(row)*width;double sum=0.;
            int hi=std::min(radius,width-1);sum=double(radius+1)*input[base];
            for(int x=1;x<=hi;++x)sum+=input[base+x];if(radius>=width)sum+=double(radius-width+1)*input[base+width-1];
            for(int x=0;x<width;++x){horizontal[base+x]=sum/double(2*radius+1);
                sum+=input[base+bounded(x+radius+1,width)]-input[base+bounded(x-radius,width)];}
        }});
        // Adjacent columns belong to the same worker: vectorize across the tile
        // and avoid cross-core cache-line contention from one-column jobs.
        int tiles=(width+31)/32;
        pool.run(planes*tiles,1,[&](int begin,int end){for(int tile=begin;tile<end;++tile){
            int plane=tile/tiles,x=(tile%tiles)*32,columns=std::min(32,width-x),hi=std::min(radius,height-1);size_t base=size_t(plane)*n+x;
            double sums[32];for(int c=0;c<columns;++c)sums[c]=double(radius+1)*horizontal[base+c];
            for(int y=1;y<=hi;++y)for(int c=0;c<columns;++c)sums[c]+=horizontal[base+size_t(y)*width+c];
            if(radius>=height)for(int c=0;c<columns;++c)sums[c]+=double(radius-height+1)*horizontal[base+size_t(height-1)*width+c];
            for(int y=0;y<height;++y){size_t enter=base+size_t(bounded(y+radius+1,height))*width,leave=base+size_t(bounded(y-radius,height))*width;
                for(int c=0;c<columns;++c)output[base+size_t(y)*width+c]=sums[c]/double(2*radius+1);
                for(int c=0;c<columns;++c)sums[c]+=horizontal[enter+c]-horizontal[leave+c];}
        }});
    }
    void run(const float* rgb,const float* first,float* targets,int radius){
        if(radius>0){
            pool.run(9*n,256,[&](int b,int e){for(int j=b;j<e;++j)guidance_product(j%n,j/n,n,rgb,raw[j]);});box(raw.data(),means.data(),9,radius);
            pool.run(n,256,[&](int b,int e){for(int p=b;p<e;++p)guidance_inverse(p,n,means.data(),info.data());});
            pool.run(4*channels*n,256,[&](int b,int e){for(int j=b;j<e;++j){int plane=j/n,p=j%n,k=plane%4,ch=plane/4;double v=source_channel(p,ch,n,layers,first);raw[j]=k?v*double(rgb[(k-1)*n+p]):v;}});
            box(raw.data(),means.data(),4*channels,radius);
            pool.run(channels*n,256,[&](int b,int e){for(int j=b;j<e;++j)coefficients(j%n,j/n,n,means.data(),info.data(),raw.data());});
            box(raw.data(),means.data(),4*channels,radius);
        }
        pool.run(n,256,[&](int b,int e){for(int p=b;p<e;++p){targets[p]=1.;
            for(int l=1;l<layers;++l)targets[l*n+p]=radius?filtered_value(p,l-1,n,rgb,means.data()):first[l*n+p];
            for(int r=0;r<3;++r)targets[(layers+r)*n+p]=(smooth&&radius)?filtered_value(p,layers-1+r,n,rgb,means.data()):first[(layers+r)*n+p];
        }});
    }
};
}
