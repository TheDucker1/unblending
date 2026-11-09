#pragma once
#include "common.hpp"
#include "../blend.hpp"
#include "float_blend.hpp"
#include <type_traits>

namespace configured {
template<class T,int Mode> HD inline T blend(T s,T d) {
    if constexpr(std::is_same_v<T,float>)return firstpass32::blend(s,d,static_cast<firstpass32::BlendMode>(Mode));
    else return firstpass::blend(s,d,static_cast<firstpass::BlendMode>(Mode));
}
template<class T,int Mode> HD inline T grad_s(T s,T d) {
    if constexpr(std::is_same_v<T,float>)return firstpass32::blend_grad_s(s,d,static_cast<firstpass32::BlendMode>(Mode));
    else return firstpass::blend_grad_s(s,d,static_cast<firstpass::BlendMode>(Mode));
}
template<class T,int Mode> HD inline T grad_d(T s,T d) {
    if constexpr(std::is_same_v<T,float>)return firstpass32::blend_grad_d(s,d,static_cast<firstpass32::BlendMode>(Mode));
    else return firstpass::blend_grad_d(s,d,static_cast<firstpass::BlendMode>(Mode));
}
template<class T> HD inline T clip(T x){return x<T(0)?T(0):(x>T(1)?T(1):x);}
template<class T> HD inline T root(T x){if constexpr(std::is_same_v<T,float>)return sqrtf(x);else return sqrt(x);}

template<class Profile,class T,bool Refinement=false,bool Diagonal=false> struct Program {
    static constexpr int L=Profile::layers,N=4*L,R=Profile::opaque_so?3:4;
    using Prefix=T[L][R];
    template<int I=1> HD static void forward_layers(const T* x,Prefix& prefix) {
        if constexpr(I<L) {
            T a=x[I];
            if constexpr(Profile::opaque_so) {
                #pragma unroll
                for(int r=0;r<3;++r){T d=prefix[I-1][r];prefix[I][r]=a*blend<T,Profile::mode[I]>(x[L+3*I+r],d)+(T(1)-a)*d;}
            }else{
                T d=prefix[I-1][3];
                T A=T(Profile::X[I])*a*d+T(Profile::Y[I])*a*(T(1)-d)+T(Profile::Z[I])*d*(T(1)-a);
                prefix[I][3]=A;
                #pragma unroll
                for(int r=0;r<3;++r){T s=x[L+3*I+r],t=prefix[I-1][r];
                    T p=blend<T,Profile::mode[I]>(s,t)*a*d+T(Profile::Y[I])*a*(T(1)-d)*s+T(Profile::Z[I])*d*(T(1)-a)*t;
                    prefix[I][r]=A>T(1e-12)?p/A:p;}
            }
            forward_layers<I+1>(x,prefix);
        }
    }
    HD static void forward(const T* x,Prefix& prefix) {
        #pragma unroll
        for(int r=0;r<3;++r)prefix[0][r]=x[L+r];
        if constexpr(!Profile::opaque_so)prefix[0][3]=x[0];
        forward_layers(x,prefix);
    }
    HD static T energy(const Layer* model,const T* x) {
        T E=T(0);
        #pragma unroll
        for(int l=0;l<L;++l){T delta[3],v[3]{};
            #pragma unroll
            for(int r=0;r<3;++r)delta[r]=x[L+3*l+r]-T(model[l].mean[r]);
            #pragma unroll
            for(int r=0;r<3;++r){
                if constexpr(Diagonal)v[r]=T(model[l].inverse[4*r])*delta[r];
                else {
                #pragma unroll
                for(int s=0;s<3;++s)v[r]+=T(model[l].inverse[3*r+s])*delta[s];}}
            E+=x[l]*(delta[0]*v[0]+delta[1]*v[1]+delta[2]*v[2]+T(Refinement?0.:.01));}
        return E;
    }
    HD static void residual(const Prefix& prefix,const T* target,T* c) {
        #pragma unroll
        for(int r=0;r<3;++r)c[r]=prefix[L-1][r]-target[r];
        c[3]=Profile::opaque_so?T(0):prefix[L-1][3]-T(1);
    }
    // A policy either records gradients or applies a complete simultaneous step.
    // Lower adjoints are finalized from old layer values before the policy runs.
    template<int I=L-1,class Policy> HD static void reverse(const Layer* model,T* x,
        const Prefix& prefix,T* adjoint,Policy& policy) {
        T a=x[I],gc[3],ga=T(0);
        if constexpr(I==0){
            #pragma unroll
            for(int r=0;r<3;++r)gc[r]=adjoint[r];
            if constexpr(!Profile::opaque_so)ga=adjoint[3];
        }else if constexpr(Profile::opaque_so){
            #pragma unroll
            for(int r=0;r<3;++r){T s=x[L+3*I+r],d=prefix[I-1][r],v=adjoint[r];
                ga+=v*(blend<T,Profile::mode[I]>(s,d)-d);
                gc[r]=v*a*grad_s<T,Profile::mode[I]>(s,d);
                adjoint[r]=v*(a*grad_d<T,Profile::mode[I]>(s,d)+T(1)-a);}
        }else{
            T d=prefix[I-1][3],A=prefix[I][3],va=adjoint[3];
            T da=T(Profile::X[I])*d+T(Profile::Y[I])*(T(1)-d)-T(Profile::Z[I])*d;
            T dd=T(Profile::X[I])*a-T(Profile::Y[I])*a+T(Profile::Z[I])*(T(1)-a);
            T previous_alpha=va*dd;ga=va*da;
            #pragma unroll
            for(int r=0;r<3;++r){T s=x[L+3*I+r],t=prefix[I-1][r],B=prefix[I][r],v=adjoint[r];
                T f=blend<T,Profile::mode[I]>(s,t);
                T ds=a*d*grad_s<T,Profile::mode[I]>(s,t)+T(Profile::Y[I])*a*(T(1)-d);
                T dt=a*d*grad_d<T,Profile::mode[I]>(s,t)+T(Profile::Z[I])*d*(T(1)-a);
                T pa=f*d+T(Profile::Y[I])*(T(1)-d)*s-T(Profile::Z[I])*d*t;
                T pd=f*a-T(Profile::Y[I])*a*s+T(Profile::Z[I])*(T(1)-a)*t;
                // Match the forward branch at zero alpha; the original divides
                // by A unconditionally, so its singular derivative is not used.
                ga+=v*(A>T(1e-12)?(pa-B*da)/A:pa);
                previous_alpha+=v*(A>T(1e-12)?(pd-B*dd)/A:pd);
                gc[r]=v*(A>T(1e-12)?ds/A:ds);
                adjoint[r]=v*(A>T(1e-12)?dt/A:dt);}
            adjoint[3]=previous_alpha;
        }
        T delta[3],v[3]{};
        #pragma unroll
        for(int r=0;r<3;++r)delta[r]=x[L+3*I+r]-T(model[I].mean[r]);
        #pragma unroll
        for(int r=0;r<3;++r){
            if constexpr(Diagonal)v[r]=T(model[I].inverse[4*r])*delta[r];
            else {
            #pragma unroll
            for(int s=0;s<3;++s)v[r]+=T(model[I].inverse[3*r+s])*delta[s];}}
        ga+=delta[0]*v[0]+delta[1]*v[1]+delta[2]*v[2]+T(Refinement?0.:.01);
        #pragma unroll
        for(int r=0;r<3;++r)gc[r]+=T(2)*a*v[r];
        policy.template apply<I>(x,ga,gc);
        if constexpr(I>0)reverse<I-1>(model,x,prefix,adjoint,policy);
    }
    struct Gradient {
        T* g;
        template<int I> HD void apply(T*,T a,const T* c){
            g[I]=Profile::opaque&&I==0?T(0):a;
            #pragma unroll
            for(int r=0;r<3;++r)g[L+3*I+r]=c[r];
        }
    };
    HD static T evaluate(const Layer* model,T* x,const T* target,const T* lambda,T rho,T* c,T* gradient) {
        Prefix prefix;forward(x,prefix);residual(prefix,target,c);
        T E=energy(model,x),adjoint[4];
        #pragma unroll
        for(int r=0;r<4;++r){adjoint[r]=rho*c[r]-lambda[r];E+=T(.5)*rho*c[r]*c[r]-lambda[r]*c[r];}
        if(gradient){Gradient g{gradient};reverse(model,x,prefix,adjoint,g);}
        return E;
    }
    struct Adam {
        T *m,*v; T scale,epsilon;
        HD void update(int j,T* x,T g){
            m[j]=T(.9)*m[j]+T(.1)*g;v[j]=T(.999)*v[j]+T(.001)*g*g;
            x[j]=clip(x[j]-scale*m[j]/(root(v[j])+epsilon));
        }
        template<int I> HD void apply(T* x,T ga,const T* gc){
            if constexpr(!(Profile::opaque&&I==0))update(I,x,ga);
            #pragma unroll
            for(int r=0;r<3;++r)update(L+3*I+r,x,gc[r]);
        }
    };
    template<bool Fused=true> HD static State solve(const Layer* model,const T* target,T* x,const Options& opt) {
        #pragma unroll
        for(int l=0;l<L;++l){x[l]=Profile::opaque&&l==0?T(1):T(.5);
            #pragma unroll
            for(int r=0;r<3;++r)x[L+3*l+r]=T(model[l].mean[r]);}
        T m[N]{},v[N]{},lambda[4]{},rho=T(opt.rho),c[4];
        for(int stage=0,done=0;done<opt.rounds;++stage){
            T bc1=T(1),bc2=T(1);
            #pragma unroll
            for(int j=0;j<N;++j)m[j]=v[j]=T(0);
            for(int k=0;k<opt.period&&done<opt.rounds;++k,++done){
                bc1*=T(.9);bc2*=T(.999);
                T correction=root(T(1)-bc2);
                T scale=T(opt.step)*(T(1)-T(.95)*T(k)/T(opt.period))*correction/(T(1)-bc1);
                Adam update{m,v,scale,T(1e-8)*correction};
                if constexpr(Fused){Prefix prefix;forward(x,prefix);residual(prefix,target,c);T adjoint[4];
                    #pragma unroll
                    for(int r=0;r<4;++r)adjoint[r]=rho*c[r]-lambda[r];
                    reverse(model,x,prefix,adjoint,update);
                }else{T g[N];evaluate(model,x,target,lambda,rho,c,g);
                    #pragma unroll
                    for(int j=0;j<N;++j)if(!(Profile::opaque&&j==0))update.update(j,x,g[j]);}
            }
            Prefix prefix;forward(x,prefix);residual(prefix,target,c);
            #pragma unroll
            for(int r=0;r<4;++r)lambda[r]-=rho*c[r];
            rho=rho*T(opt.growth)>T(10000)?T(10000):rho*T(opt.growth);
        }
        T E=energy(model,x);T norm=root(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]+c[3]*c[3]);
        return {float(norm),float(E)};
    }
};
}
