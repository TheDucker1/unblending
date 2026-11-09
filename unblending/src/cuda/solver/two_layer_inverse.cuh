#pragma once
#include "normal_blend_refinement.cuh"

namespace two_inverse {
template<class T>struct Intervals {T lower[2],upper[2];int count=0;};
template<class T>HD void add_interval(Intervals<T>& result,T lo,T hi){
    lo=configured::clip(lo);hi=configured::clip(hi);if(lo<=hi){result.lower[result.count]=lo;result.upper[result.count]=hi;++result.count;}
}

// Full preimage of a blend-value interval for a fixed, unit-box destination.
// All13 CPU blend functions are source-affine/piecewise-affine or monotone
// rational. Difference has two branches; clipped/constant blends have intervals.
// Retaining intervals matters: choosing one arbitrary plateau color loses the
// Gaussian minimum when RGB covariance couples its three source coordinates.
template<class T>HD Intervals<T> preimage(T d,int mode,T lo,T hi){
    Intervals<T> out;T a=general::blend(T(0),d,mode),b=general::blend(T(1),d,mode);
    if(mode!=10&&a==b){add_interval(out,T(0),T(1));return out;}
    switch(mode){
        case 0:add_interval(out,lo,hi);break;
        case 1:add_interval(out,lo/d,hi/d);break;
        case 2:add_interval(out,(lo-d)/(1-d),(hi-d)/(1-d));break;
        case 3:{T slope=d<=T(.5)?2*d:2*(1-d),offset=d<=T(.5)?T(0):2*d-1;add_interval(out,(lo-offset)/slope,(hi-offset)/slope);break;}
        case 4:add_interval(out,lo,hi>=d?T(1):hi);break;
        case 5:add_interval(out,lo<=d?T(0):lo,hi);break;
        case 6:add_interval(out,lo<=d?T(0):1-d/lo,hi>=1?T(1):1-d/hi);break;
        case 7:add_interval(out,lo<=0?T(0):(1-d)/(1-lo),hi>=d?T(1):(1-d)/(1-hi));break;
        case 8:{T lower,upper;
            if(lo<=d)lower=d>0?lo/(2*d):T(0);else lower=1-(1-lo)/(2*(1-d));
            if(hi>=d)upper=d<1?1-(1-hi)/(2*(1-d)):T(1);else upper=hi/(2*d);add_interval(out,lower,upper);break;}
        case 9:{T g=d<=T(.25)?((16*d-12)*d+4)*d:configured::root(d);
            T lower=lo<=d?(lo-d*d)/(2*d*(1-d)):(lo-2*d+g)/(2*(g-d));
            T upper=hi<=d?(hi-d*d)/(2*d*(1-d)):(hi-2*d+g)/(2*(g-d));add_interval(out,lower,upper);break;}
        case 10:{
            // Intersect each branch BEFORE unit clipping, otherwise an empty
            // branch could incorrectly become a spurious endpoint solution.
            T l=d-hi,h=d-lo;if(l<0)l=0;if(h>d)h=d;if(l<=h)add_interval(out,l,h);
            l=d+lo;h=d+hi;if(l<d)l=d;if(h>1)h=1;if(l<=h){if(out.count&&l<=out.upper[0])out.upper[0]=h;else add_interval(out,l,h);}break;}
        case 11:{T slope=1-2*d;if(slope>0)add_interval(out,(lo-d)/slope,(hi-d)/slope);else add_interval(out,(hi-d)/slope,(lo-d)/slope);break;}
        case 12:add_interval(out,lo-d,hi-d);break;
    }
    return out;
}

template<class T,class Input=T,class Output=T,bool FastInterval=false>HD int pixel(const gaussian_box::Model* model,general::Problem problem,int n,int index,const Input* rgb,const Input* alpha,const Input* background,Output* output,T tolerance,T* candidate_error=nullptr){
    T av=T(alpha[size_t(n)+index]),bg[3],target[3],goal[3];
    if(alpha[index]!=1||alpha[size_t(n)+index]<0||alpha[size_t(n)+index]>1||!std::isfinite(alpha[size_t(n)+index]))return 4;
    output[index]=alpha[index];output[size_t(n)+index]=alpha[size_t(n)+index];
    for(int r=0;r<3;++r){Input original=background[size_t(r)*n+index];bg[r]=T(original);target[r]=T(rgb[size_t(r)*n+index]);if(original<0||original>1||!std::isfinite(original)||!std::isfinite(target[r]))return 4;output[size_t(2+r)*n+index]=original;}
    const auto& m=model[1];T A=T(m.X)*av+T(m.Z)*(1-av),den=A>T(1e-12)?A:T(1),amplitude=av/den,offset=T(m.Z)*(1-av)/den;
    Intervals<T> intervals[3];bool projected=false;T budget=tolerance/T(1.7320508075688772935);
    for(int r=0;r<3;++r){
        T f0=general::blend(T(0),bg[r],m.mode),f1=general::blend(T(1),bg[r],m.mode);T lower=f0<f1?f0:f1,upper=f0>f1?f0:f1;if(m.mode==10)lower=0;
        T low=amplitude*lower+offset*bg[r],high=amplitude*upper+offset*bg[r];goal[r]=target[r]<low?low:(target[r]>high?high:target[r]);projected|=goal[r]!=target[r];
        if(amplitude==0)add_interval(intervals[r],T(0),T(1));
        else {T lo=(goal[r]-budget-offset*bg[r])/amplitude,hi=(goal[r]+budget-offset*bg[r])/amplitude;if(lo<lower)lo=lower;if(hi>upper)hi=upper;intervals[r]=preimage(bg[r],m.mode,lo,hi);}
        if(!intervals[r].count)return 2;
    }
    bool found=false;T best=INFINITY,color[3]{},chosen[3]{};
    for(int i=0;i<intervals[0].count;++i)for(int j=0;j<intervals[1].count;++j)for(int k=0;k<intervals[2].count;++k){
        T lo[3]={intervals[0].lower[i],intervals[1].lower[j],intervals[2].lower[k]},hi[3]={intervals[0].upper[i],intervals[1].upper[j],intervals[2].upper[k]};
        if(m.gray){T lower=lo[0],upper=hi[0],numerator=0,curvature=0;for(int r=1;r<3;++r){if(lo[r]>lower)lower=lo[r];if(hi[r]<upper)upper=hi[r];}if(lower>upper)continue;
            for(int r=0;r<3;++r)for(int s=0;s<3;++s){numerator+=T(m.inverse[3*r+s]*m.mean[s]);curvature+=T(m.inverse[3*r+s]);}
            T value=numerator/curvature;value=value<lower?lower:(value>upper?upper:value);for(int r=0;r<3;++r)color[r]=value;
        }else if constexpr(FastInterval){
            // A deliberate speed/energy tradeoff: remain inside the full
            // reconstruction preimage, but skip active-face QP optimization.
            // Full covariance still ranks disconnected Difference branches.
            for(int r=0;r<3;++r){T mean=T(m.mean[r]);color[r]=mean<lo[r]?lo[r]:(mean>hi[r]?hi[r]:mean);}
        }else if(gaussian_box::solve_interval(m,lo,hi,color)<0)continue;
        T cost=general::GaussianPrior::evaluate(static_cast<const general::Layer&>(m),color,static_cast<T*>(nullptr));if(!found||cost<best){best=cost;found=true;for(int r=0;r<3;++r)chosen[r]=color[r];}
    }
    int status=projected?5:0;
    if(!found){
        if(!m.gray)return 2;
        // No gray color in the requested RGB band. A bounded device-only grid
        // supplies a flagged best-effort result, not a feasibility certificate.
        T closest=INFINITY;
        #pragma unroll 1
        for(int sample=0;sample<=128;++sample){T value=T(sample)/T(128),error=0;for(int r=0;r<3;++r){T delta=amplitude*general::blend(value,bg[r],m.mode)+offset*bg[r]-target[r];error+=delta*delta;}
            if(error<closest){closest=error;for(int r=0;r<3;++r)chosen[r]=value;}}
        status=6;
    }
    T error=0;for(int r=0;r<3;++r){output[size_t(5+r)*n+index]=chosen[r];T delta=amplitude*general::blend(chosen[r],bg[r],m.mode)+offset*bg[r]-goal[r];error+=delta*delta;}
    if(candidate_error)*candidate_error=error;
    if(status!=6&&configured::root(error)>tolerance+T(sizeof(T)==4?5e-6:1e-10))status=3;
    if(model[0].gray&&(bg[0]!=bg[1]||bg[1]!=bg[2]))status=7;
    return status;
}
template<class T,bool FastInterval=false>__global__ void kernel(const gaussian_box::Model* model,general::Problem problem,int n,const T* rgb,const T* alpha,const T* background,T* output,int* status,T tolerance){
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index<n)status[index]=pixel<T,T,T,FastInterval>(model,problem,n,index,rgb,alpha,background,output,tolerance);
}

// CPU API images/targets are double. Retain them and score the FP32 candidate
// using the actual double forward function before selecting rare GPU repairs.
// In particular a rounded background must not silently select a different
// Dodge/Burn branch. Double foreground output also avoids losing the repair
// again when a steep inverse requires a color between adjacent FP32 values.
template<bool FastInterval=false,bool FastGuard=false>HD inline int mixed_initial(const gaussian_box::Model* model,general::Problem problem,int n,int index,const double* rgb,const double* alpha,const double* background,double* output,double tolerance,int& repair){
    const auto& descriptor=model[1];
    // For opaque source-over/Plus and continuous non-Dodge/Burn blends, all
    // colors are bounded, blend slopes are <=4 and alpha geometry is stable.
    // Reserve 8e-6 for FP32 forward/target rounding, twice in the candidate
    // band. Near-singular composition and discontinuous rational cases keep
    // the original double forward validation below. No CPU/GPU transfer.
    bool inexpensive_guard=FastGuard&&tolerance>=4e-5&&descriptor.Z==1&&(descriptor.X==1||descriptor.X==2)&&descriptor.mode!=6&&descriptor.mode!=7;
    float margin=8e-6f,candidate_tolerance=float(tolerance)-(inexpensive_guard?2*margin:0),error32=INFINITY;
    int status=pixel<float,double,double,FastInterval>(model,problem,n,index,rgb,alpha,background,output,candidate_tolerance,&error32);repair=0;
    if(inexpensive_guard){float allowance=float(tolerance)-margin;repair=!std::isfinite(error32)||error32>allowance*allowance||status==2||status==4||status==6||status==7;return status;}
    if(status==2||status==4||status==6){repair=1;return status;}
    const auto& m=model[1];double a=alpha[size_t(n)+index],A=m.X*a+m.Z*(1-a),den=A>1e-12?A:1,amplitude=a/den,offset=m.Z*(1-a)/den,error=0;bool projected=false;
    for(int r=0;r<3;++r){double d=background[size_t(r)*n+index],f0=general::blend(0.,d,m.mode),f1=general::blend(1.,d,m.mode),low=f0<f1?f0:f1,high=f0>f1?f0:f1;if(m.mode==10)low=0;
        low=amplitude*low+offset*d;high=amplitude*high+offset*d;double target=rgb[size_t(r)*n+index],goal=target<low?low:(target>high?high:target);projected|=goal!=target;
        double color=output[size_t(5+r)*n+index],delta=amplitude*general::blend(color,d,m.mode)+offset*d-goal;error+=delta*delta;
    }
    repair=!std::isfinite(error)||error>(tolerance+2e-6)*(tolerance+2e-6);
    if(status==0||status==3||status==5)status=projected?5:0;
    return status;
}
template<bool FastInterval=false,bool FastGuard=false>__global__ void mixed_kernel(const gaussian_box::Model* model,general::Problem problem,int n,const double* rgb,const double* alpha,const double* background,double* output,int* status,int* repair,double tolerance){
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index<n)status[index]=mixed_initial<FastInterval,FastGuard>(model,problem,n,index,rgb,alpha,background,output,tolerance,repair[index]);
}
template<bool FastInterval=false>__global__ void repair_kernel(const gaussian_box::Model* model,general::Problem problem,int n,const double* rgb,const double* alpha,const double* background,double* output,int* status,const int* repair,double tolerance){
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index<n&&repair[index])status[index]=pixel<double,double,double,FastInterval>(model,problem,n,index,rgb,alpha,background,output,tolerance);
}
}
