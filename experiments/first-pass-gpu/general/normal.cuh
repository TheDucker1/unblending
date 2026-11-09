#pragma once
#include "program.cuh"
#include "box.cuh"

namespace normal_refinement {
using Model=gaussian_box::Model;

// Symmetric positive-definite3x3 solve, scaled to avoid determinant overflow.
// The general active-face extension will use the appropriate principal block.
template<class T>HD bool solve3(const T* K,T sigma,const T* rhs,T* result){
    T scale=K[0]+K[3]+K[5]+T(3)*sigma;if(!(scale>0)||!std::isfinite(scale))return false;
    T a=(K[0]+sigma)/scale,b=K[1]/scale,c=K[2]/scale,d=(K[3]+sigma)/scale,e=K[4]/scale,f=(K[5]+sigma)/scale;
    T A=d*f-e*e,B=c*e-b*f,C=b*e-c*d,D=a*f-c*c,E=b*c-a*e,F=a*d-b*b,det=a*A+b*B+c*C;
    if(!(det>0)||!std::isfinite(T(1)/det))return false;
    T factor=T(1)/(det*scale);result[0]=factor*(A*rhs[0]+B*rhs[1]+C*rhs[2]);result[1]=factor*(B*rhs[0]+D*rhs[1]+E*rhs[2]);result[2]=factor*(C*rhs[0]+E*rhs[1]+F*rhs[2]);
    return std::isfinite(result[0])&&std::isfinite(result[1])&&std::isfinite(result[2]);
}
template<class T>HD T norm3(const T* v){return configured::root(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);}

// Fixed-alpha/full-covariance Normal refinement, interior solution. Flags
// explicitly identify pixels needing bound-face or degenerate treatment;
// this prototype never executes a hidden CPU fallback.
// With tolerance zero this is the exact equality-constrained convex QP.
// Positive tolerance is an explicitly reported numerical RGB residual budget;
// alpha targets, original image and all independent checks remain unchanged.
template<class T>HD int pixel(const Model* model,general::Problem p,int n,int index,const T* target,const T* alpha,
    const T* background,T* output,T* scratch,T tolerance,T* seed=nullptr){
    int L=p.layers;auto a=[&](int l){return alpha[size_t(l)*n+index];};auto w=[&](int l)->T&{return scratch[size_t(l)*n+index];};
    if(seed)for(int r=0;r<3;++r)seed[size_t(r)*n+index]=0;
    for(int l=0;l<L;++l){T value=a(l);if(value<0||value>1||!std::isfinite(value)||(p.opaque&&l==0&&value!=1))return 4;output[size_t(l)*n+index]=value;}
    T suffix=1;
    if(model[0].opaque_source_over){for(int l=L-1;l>0;--l){w(l)=a(l)*suffix;suffix*=1-a(l);}w(0)=suffix;}
    else {w(0)=a(0);for(int l=1;l<L;++l){T s=a(l),d=w(l-1);const auto& m=model[l];w(l)=T(m.X)*s*d+T(m.Y)*s*(1-d)+T(m.Z)*d*(1-s);}
        for(int l=L-1;l>0;--l){T s=a(l),d=w(l-1),A=w(l),den=A>T(1e-12)?A:T(1);const auto& m=model[l];
            w(l)=s*(d+T(m.Y)*(1-d))/den*suffix;suffix*=T(m.Z)*d*(1-s)/den;}w(0)=suffix;}
    // One observed color has no optimization freedom. In particular Normal
    // with one fixed background and one foreground determines foreground RGB
    // directly, for ANY color prior/covariance. Avoid a Gram/root solve here.
    if(L==1||(L==2&&p.smooth)){
        bool projected=false;
        for(int r=0;r<3;++r){T desired=target[size_t(r)*n+index];
            if(L==1){T color=p.smooth?background[size_t(r)*n+index]:configured::clip(desired);output[size_t(L+r)*n+index]=color;projected|=p.smooth?std::abs(color-desired)>tolerance:(desired<0||desired>1);}
            else {T bg=background[size_t(r)*n+index];output[size_t(L+r)*n+index]=bg;
                T raw=w(1)!=0?(desired-w(0)*bg)/w(1):T(model[1].mean[r]);T color=configured::clip(raw);output[size_t(L+3+r)*n+index]=color;
                projected|=w(1)!=0?(raw<0||raw>1):std::abs(w(0)*bg-desired)>tolerance;}}
        return projected?5:0;
    }
    T K[6]{},rhs[3];for(int r=0;r<3;++r)rhs[r]=-target[size_t(r)*n+index];
    for(int l=0;l<L;++l){T weight=w(l),value=a(l);bool fixed=p.smooth&&l==0;const auto& m=model[l];
        if(!std::isfinite(weight))return 4;
        for(int r=0;r<3;++r)rhs[r]+=weight*(fixed?background[size_t(r)*n+index]:T(m.mean[r]));
        if(fixed||weight==0)continue;if(!(value>0))return 2;T scale=weight*weight/(T(2)*value);
        const int rows[6]={0,0,0,1,1,2},cols[6]={0,1,2,1,2,2};for(int k=0;k<6;++k)K[k]+=scale*T(m.covariance[3*rows[k]+cols[k]]);
    }
    T dual[3]{};T length=norm3(rhs);
    if(length>tolerance){
        if(tolerance==0){if(!solve3(K,T(0),rhs,dual))return 2;}
        else {
            // Gershgorin upper eigenvalue bounds the scalar root bracket.
            T high=K[0]+std::abs(K[1])+std::abs(K[2]);T v=K[3]+std::abs(K[1])+std::abs(K[4]);if(v>high)high=v;
            v=K[5]+std::abs(K[2])+std::abs(K[4]);if(v>high)high=v;high=tolerance*high/(length-tolerance);T low=0;
            #pragma unroll 1
            for(int iteration=0;iteration<28;++iteration){T mid=(low+high)*T(.5);if(!solve3(K,mid,rhs,dual))return 2;
                T residual=mid*norm3(dual);if(residual<tolerance)low=mid;else high=mid;}
            if(!solve3(K,(low+high)*T(.5),rhs,dual))return 2;
        }
    }
    if(seed)for(int r=0;r<3;++r)seed[size_t(r)*n+index]=dual[r];
    int status=0;for(int l=0;l<L;++l){T value=a(l),weight=w(l);const auto& m=model[l];
        for(int r=0;r<3;++r){T color;
            if(p.smooth&&l==0)color=background[size_t(r)*n+index];
            else if(value==0&&weight==0)color=configured::clip(T(m.mean[r]));
            else {color=T(m.mean[r]);if(weight!=0){T v=0;for(int s=0;s<3;++s)v+=T(m.covariance[3*r+s])*dual[s];color-=weight/(T(2)*value)*v;}}
            if(color<0||color>1||!std::isfinite(color))status=1;output[size_t(L+3*l+r)*n+index]=color;}}
    return status;
}

template<class T>__global__ void kernel(const Model* model,general::Problem p,int n,const T* target,const T* alpha,const T* background,T* output,T* scratch,int* status,T tolerance){
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index<n)status[index]=pixel(model,p,n,index,target,alpha,background,output,scratch,tolerance,scratch+size_t(p.layers)*n);
}
}
