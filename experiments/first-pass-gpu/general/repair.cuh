#pragma once
#include "program.cuh"
#include <limits>

namespace general {
#ifdef __CUDACC__
#define REPAIR_NOINLINE __noinline__
#else
#define REPAIR_NOINLINE
#endif
// Full output-RGBA Jacobian in one reverse sweep. RGB composition has no
// cross-channel color dependencies, while all RGB channels depend on alpha.
// Store four rows of 4*layers entries; unused cross-channel entries are zero.
template<class T,class Model>HD void rgba_jacobian(const Model* model,const Problem& p,
    View<const T> x,View<const T> prefix,View<T> jac,bool sparse=false){
    int L=p.layers,N=4*L;if(!sparse)for(int k=0;k<4*N;++k)jac[k]=0;
    T color[3]={1,1,1},opacity[3]{},final_alpha=1;
    for(int l=L-1;l>0;--l){
        const auto& m=model[l];T a=x[l],d=prefix[4*(l-1)+3],A=prefix[4*l+3];
        T da=T(m.X)*d+T(m.Y)*(1-d)-T(m.Z)*d;
        T dd=T(m.X)*a-T(m.Y)*a+T(m.Z)*(1-a);
        jac[3*N+l]=final_alpha*da;final_alpha*=dd;
        for(int r=0;r<3;++r){
            T s=x[L+3*l+r],c=prefix[4*(l-1)+r],out=prefix[4*l+r];
            T f=blend(s,c,m.mode),ds=a*d*grad_s(s,c,m.mode)+T(m.Y)*a*(1-d);
            T dc=a*d*grad_d(s,c,m.mode)+T(m.Z)*d*(1-a);
            T pa=f*d+T(m.Y)*(1-d)*s-T(m.Z)*d*c;
            T pd=f*a-T(m.Y)*a*s+T(m.Z)*(1-a)*c;
            if(A>T(1e-12)){ds/=A;dc/=A;pa=(pa-out*da)/A;pd=(pd-out*dd)/A;}
            jac[r*N+l]=opacity[r]*da+color[r]*pa;
            jac[r*N+L+3*l+r]=color[r]*ds;
            opacity[r]=opacity[r]*dd+color[r]*pd;color[r]*=dc;
        }
    }
    jac[3*N]=final_alpha;
    for(int r=0;r<3;++r){jac[r*N]=opacity[r];jac[r*N+L+r]=color[r];}
}

// With alpha held fixed, the paper's scalar blend class couples no distinct
// RGB channels. One reverse sweep therefore gives every nonzero color partial
// of final RGB, plus the independent final-alpha derivatives. Full covariance
// affects the energy, but cannot introduce cross-channel composition partials.
template<class T,class Model>HD void composition_jacobian(const Model* model,const Problem& p,
    View<const T> x,View<const T> prefix,View<T> jac){
    int L=p.layers;T color[3]={1,1,1},alpha=1;
    for(int l=L-1;l>0;--l){const auto& m=model[l];T a=x[l],d=prefix[4*(l-1)+3],A=prefix[4*l+3];
        jac[l]=alpha*(T(m.X)*d+T(m.Y)*(1-d)-T(m.Z)*d);
        alpha*=T(m.X)*a-T(m.Y)*a+T(m.Z)*(1-a);
        for(int r=0;r<3;++r){T s=x[L+3*l+r],c=prefix[4*(l-1)+r];
            T ds=a*d*grad_s(s,c,m.mode)+T(m.Y)*a*(1-d);
            T dc=a*d*grad_d(s,c,m.mode)+T(m.Z)*d*(1-a);
            jac[L+3*l+r]=color[r]*(A>T(1e-12)?ds/A:ds);
            color[r]*=A>T(1e-12)?dc/A:dc;
        }
    }
    jac[0]=alpha;for(int r=0;r<3;++r)jac[L+r]=color[r];
}

// Solve a regularized 3x3 Gram system. Diagonal systems (no grayscale layer)
// need three divisions; coupled grayscale directions use pivoted elimination.
template<class T>HD void repair_system(T* gram,const T* rhs,T* step){
    T scale=gram[0]+gram[4]+gram[8];for(int r=0;r<3;++r)step[r]=0;if(!(scale>0))return;
    if(gram[1]==0&&gram[2]==0&&gram[5]==0){for(int r=0;r<3;++r)if(gram[4*r]>0)step[r]=rhs[r]/gram[4*r];return;}
    T a[12];T ridge=scale*T(sizeof(T)==4?1e-6:1e-12);
    for(int r=0;r<3;++r){for(int s=0;s<3;++s)a[4*r+s]=gram[3*r+s]+(r==s?ridge:T(0));a[4*r+3]=rhs[r];}
    for(int k=0;k<3;++k){int pivot=k;for(int r=k+1;r<3;++r)if(std::abs(a[4*r+k])>std::abs(a[4*pivot+k]))pivot=r;
        if(pivot!=k)for(int s=k;s<4;++s){T v=a[4*k+s];a[4*k+s]=a[4*pivot+s];a[4*pivot+s]=v;}
        if(!(std::abs(a[4*k+k])>0)||!std::isfinite(a[4*k+k]))return;
        for(int r=k+1;r<3;++r){T f=a[4*r+k]/a[4*k+k];for(int s=k;s<4;++s)a[4*r+s]-=f*a[4*k+s];}}
    for(int r=2;r>=0;--r){T v=a[4*r+3];for(int s=r+1;s<3;++s)v-=a[4*r+s]*step[s];step[r]=v/a[4*r+r];}
}

template<class T,class Prior,class Model=Layer>struct Repair {
    using P=Program<T,Prior,Model>;
    // One outlined value routine prevents repeated full runtime-blend program
    // copies in each correction/backtracking path. All state remains device-side.
    HD static REPAIR_NOINLINE T evaluate_point(const Model* model,const Problem& p,View<T> point,const T* target,
        View<const T> alpha,View<T> lambda,View<T> prefix,View<T> residual,View<T> scratch,T& energy){
        P::evaluate(model,p,{point.data,point.stride},target,alpha,{lambda.data,lambda.stride},T(0),prefix,residual,scratch,false,&energy);
        T norm=0;int C=p.constraints(model);for(int r=0;r<C;++r)norm+=residual[r]*residual[r];return norm;
    }
    HD static bool run(const Model* model,const Problem& p,const T* target,View<const T> alpha,const T* bg,
        View<const T> original,View<T> x,View<T> trial,View<T> prefix,View<T> jac,View<T> residual,View<T> lambda,
        T original_energy,T original_norm,unsigned* values=nullptr){
        int L=p.layers,N=4*L,C=p.constraints(model);if(!std::isfinite(original_norm)||original_norm>T(.01*.01))return false;
        auto cv=[](View<T> v){return View<const T>{v.data,v.stride};};
        for(int k=0;k<N;++k)x[k]=original[k];for(int r=0;r<C;++r)lambda[r]=0;
        if(p.refinement){for(int l=0;l<L;++l){T value=alpha[l];if(value<0||value>1||(p.opaque&&l==0&&value!=1))return false;x[l]=value;}}
        for(int l=0;l<L;++l)if(model[l].gray){if(p.refinement&&p.smooth&&l==0){if(bg[0]!=bg[1]||bg[1]!=bg[2])return false;}
            else {T mean=(x[L+3*l]+x[L+3*l+1]+x[L+3*l+2])/T(3);for(int r=0;r<3;++r)x[L+3*l+r]=mean;}}
        T energy=0;
        auto evaluate=[&](View<T> point){if(values)++*values;return evaluate_point(model,p,point,target,alpha,lambda,prefix,residual,jac,energy);};
        T norm=evaluate(x);
        #pragma unroll 1
        for(int iteration=0;iteration<24&&norm>T(1e-10);++iteration){
            // Final alpha is color-independent. Refined alphas were already
            // set to their actual equality targets, with opaque bounds checked.
            if(!p.refinement&&std::abs(residual[3])>T(1e-7)){
                composition_jacobian(model,p,cv(x),cv(prefix),jac);T sum=0,R=residual[3];
                for(int l=0;l<L;++l)if(!(p.opaque&&l==0)){T d=-R*jac[l];if(!((x[l]<=0&&d<0)||(x[l]>=1&&d>0)))sum+=jac[l]*jac[l];}
                if(sum>0){for(int k=0;k<N;++k)trial[k]=x[k];for(int l=0;l<L;++l)if(!(p.opaque&&l==0)){T v=x[l]-R*jac[l]/sum;trial[l]=v<0?0:(v>1?1:v);}
                    T next=evaluate(trial);if(next<norm){for(int k=0;k<N;++k)x[k]=trial[k];norm=next;}else norm=evaluate(x);}
            }
            composition_jacobian(model,p,cv(x),cv(prefix),jac);T gram[9]{},rhs[3],step[3];for(int r=0;r<3;++r)rhs[r]=-residual[r];
            for(int l=0;l<L;++l){if(p.refinement&&p.smooth&&l==0)continue;
                if(model[l].gray){T j[3];for(int r=0;r<3;++r)j[r]=jac[L+3*l+r];for(int r=0;r<3;++r)for(int s=0;s<3;++s)gram[3*r+s]+=j[r]*j[s]/T(3);}
                else for(int r=0;r<3;++r){int k=L+3*l+r;T j=jac[k],d=j*rhs[r];if(!((x[k]<=0&&d<0)||(x[k]>=1&&d>0)))gram[4*r]+=j*j;}}
            repair_system(gram,rhs,step);
            // The Gram system is small even for hundreds of layers. Backtrack
            // on the actual constraints after enforcing color/gray bounds.
            bool accepted=false;T amount=1;
            #pragma unroll 1
            for(int backtrack=0;backtrack<12;++backtrack){for(int k=0;k<N;++k)trial[k]=x[k];
                for(int l=0;l<L;++l){if(p.refinement&&p.smooth&&l==0)continue;T gray=0;if(model[l].gray)for(int r=0;r<3;++r)gray+=jac[L+3*l+r]*step[r]/T(3);
                    for(int r=0;r<3;++r){int k=L+3*l+r;T v=x[k]+amount*(model[l].gray?gray:jac[k]*step[r]);trial[k]=v<0?0:(v>1?1:v);}}
                T next=evaluate(trial);if(std::isfinite(next)&&next<norm){for(int k=0;k<N;++k)x[k]=trial[k];norm=next;accepted=true;break;}amount*=T(.5);}
            if(!accepted){norm=evaluate(x);break;}
        }
        norm=evaluate(x);T budget=T(.00005)+T(.005)*std::abs(original_energy);
        // This is a candidate correction, not a new objective or tolerance.
        // Never replace an incumbent by a materially more expensive result.
        return std::isfinite(energy)&&std::isfinite(norm)&&norm<original_norm&&energy<=original_energy+budget;
    }
};
#undef REPAIR_NOINLINE
}
