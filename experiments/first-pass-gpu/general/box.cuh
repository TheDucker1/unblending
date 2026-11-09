#pragma once
#include "program.cuh"
#include <limits>

namespace gaussian_box {
struct Model:general::Layer {
    double covariance[9];
    // Embedded inverses of the seven nonempty principal precision blocks.
    // Configuration data, shared across pixels and reusable across images.
    double face_inverse[8][9]{};
    int opaque_source_over=0;
};
template<class T>HD T gradient(const Model& m,const T* color,const T* linear,int r){
    // normal_models has already symmetrized S on CPU once per configuration.
    // Repeating double add/multiply here throttles FP32 GPU face tests.
    T result=linear[r];for(int s=0;s<3;++s)result+=T(m.inverse[3*r+s])*(color[s]-T(m.mean[s]));return result;
}
template<class T>HD bool face(const Model& m,const T* linear,int mask,const T* bounds,T* color,int force=0){
    T rhs[3];for(int r=0;r<3;++r){rhs[r]=linear[r];color[r]=bounds[r];
        for(int s=0;s<3;++s)if(!(mask&(1<<s)))rhs[r]+=T(m.inverse[3*r+s])*(bounds[s]-T(m.mean[s]));}
    T tolerance=T(32)*std::numeric_limits<T>::epsilon();
    for(int r=0;r<3;++r)if(mask&(1<<r)){T value=T(m.mean[r]);for(int s=0;s<3;++s)value-=T(m.face_inverse[mask][3*r+s])*rhs[s];
        if(!std::isfinite(value)||value<-tolerance||value>T(1)+tolerance)return false;color[r]=configured::clip(value);}
    for(int r=0;r<3;++r)if(!((mask|force)&(1<<r))){T g=gradient(m,color,linear,r),scale=std::abs(linear[r]);
        // Backward-error scale for S*(color-mean)+linear. No additive1 in
        // energy units: uniformly rescaling the QP must not change its face.
        for(int s=0;s<3;++s)scale+=std::abs(T(m.inverse[3*r+s]))*(T(1)+std::abs(color[s])+std::abs(T(m.mean[s])));
        T eps=T(64)*std::numeric_limits<T>::epsilon()*scale;if((bounds[r]==0&&g<-eps)||(bounds[r]==1&&g>eps))return false;}
    return true;
}

// argmin 0.5*(color-mean)'*S*(color-mean) + linear'*color over [0,1]^3.
// Full covariance is retained. Warm clamped face first; exhaustive27faces is
// a bounded fallback for release of a wrongly active coordinate. KKT tests,
// rather than clamping an unconstrained vector, determine the valid face.
template<class T>HD int solve(const Model& m,const T* linear,T* color,int force=0,const T* forced=nullptr){
    T raw[3],bounds[3]{};int guess=7&~force;
    for(int r=0;r<3;++r){T v=T(m.mean[r]);for(int s=0;s<3;++s)v-=T(m.covariance[3*r+s])*linear[s];raw[r]=v;
        if(force&(1<<r)){bounds[r]=forced[r];guess&=~(1<<r);}else if(v<=0||v>=1){bounds[r]=v<=0?T(0):T(1);guess&=~(1<<r);}}
    if(face(m,linear,guess,bounds,color,force))return guess;
    #pragma unroll 1
    for(int code=0;code<27;++code){int digits=code,mask=0;bool allowed=true;
        for(int r=0;r<3;++r){int state=digits%3;digits/=3;if(state==0)mask|=1<<r;bounds[r]=state==2?T(1):T(0);
            if((force&(1<<r))&&(state==0||bounds[r]!=forced[r]))allowed=false;}
        if(allowed&&face(m,linear,mask,bounds,color,force))return mask;
    }
    // An SPD box problem has a minimizer. A miss is a numerical/model error,
    // reported to callers; the bounded vector is diagnostic, not a solved QP.
    for(int r=0;r<3;++r)color[r]=(force&(1<<r))?forced[r]:configured::clip(raw[r]);return -1;
}

// Gaussian minimum on a per-channel interval, including zero-width intervals.
// Used by two-layer inverse blend maps. Exact colors are equality constraints,
// so their gradient signs must not be mistaken for releasable active bounds.
template<class T>HD bool interval_face(const Model& m,const T* lower,const T* upper,int mask,const T* bounds,T* color){
    T rhs[3]{};for(int r=0;r<3;++r){color[r]=bounds[r];for(int s=0;s<3;++s)if(!(mask&(1<<s)))rhs[r]+=T(m.inverse[3*r+s])*(bounds[s]-T(m.mean[s]));}
    T tolerance=T(32)*std::numeric_limits<T>::epsilon();
    for(int r=0;r<3;++r)if(mask&(1<<r)){T value=T(m.mean[r]);for(int s=0;s<3;++s)value-=T(m.face_inverse[mask][3*r+s])*rhs[s];
        if(!std::isfinite(value)||value<lower[r]-tolerance||value>upper[r]+tolerance)return false;color[r]=value<lower[r]?lower[r]:(value>upper[r]?upper[r]:value);}
    const T zero[3]{};
    for(int r=0;r<3;++r)if(!(mask&(1<<r))&&lower[r]<upper[r]){T g=gradient(m,color,zero,r),scale=0;
        for(int s=0;s<3;++s)scale+=std::abs(T(m.inverse[3*r+s]))*(T(1)+std::abs(color[s])+std::abs(T(m.mean[s])));
        T eps=T(64)*std::numeric_limits<T>::epsilon()*scale;if((bounds[r]==lower[r]&&g<-eps)||(bounds[r]==upper[r]&&g>eps))return false;}
    return true;
}
template<class T>HD int solve_interval(const Model& m,const T* lower,const T* upper,T* color){
    int guess=0;T bounds[3];for(int r=0;r<3;++r){T mean=T(m.mean[r]);if(lower[r]>upper[r])return -1;
        bounds[r]=mean<lower[r]?lower[r]:(mean>upper[r]?upper[r]:mean);if(lower[r]<upper[r]&&mean>lower[r]&&mean<upper[r])guess|=1<<r;}
    if(interval_face(m,lower,upper,guess,bounds,color))return guess;
    #pragma unroll 1
    for(int code=0;code<27;++code){int digits=code,mask=0;bool allowed=true;
        for(int r=0;r<3;++r){int state=digits%3;digits/=3;if(state==0){mask|=1<<r;if(lower[r]==upper[r])allowed=false;}bounds[r]=state==2?upper[r]:lower[r];}
        if(allowed&&interval_face(m,lower,upper,mask,bounds,color))return mask;}
    return -1;
}
}
