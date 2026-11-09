#pragma once
#include "compositing/fixed_stack.cuh"
#include <cmath>

namespace general {
// Runtime numeric configuration. Full covariance and composition coefficients
// are never inferred from a benchmark profile or silently diagonalized.
struct Layer {
    double mean[3], inverse[9];
    int mode=0,X=1,Y=1,Z=1,gray=0;
    double quartic=0;
};
struct Problem {
    int layers;
    bool refinement=false,opaque=true,smooth=false,fixed_alpha=false;
    bool sparsity=false,minimum=true;
    double sparsity_weight=10.;
    template<class Model>HD int constraints(const Model* model)const{
        int n=3+(refinement?layers:1);for(int l=0;l<layers;++l)n+=3*model[l].gray;return n;
    }
};
// Same routines work on compact CPU arrays and coalesced [variable,pixel]
// GPU planes. Runtime layer counts do not require register arrays or a32 cap.
template<class T>struct View {
    T* data;size_t stride=1;
    HD T& operator[](int i)const{return data[size_t(i)*stride];}
};
template<class T>HD T blend(T s,T d,int mode){
    if constexpr(std::is_same_v<T,float>)return firstpass32::blend(s,d,static_cast<firstpass32::BlendMode>(mode));
    else return firstpass::blend(s,d,static_cast<firstpass::BlendMode>(mode));
}
template<class T>HD T grad_s(T s,T d,int mode){
    if constexpr(std::is_same_v<T,float>)return firstpass32::blend_grad_s(s,d,static_cast<firstpass32::BlendMode>(mode));
    else return firstpass::blend_grad_s(s,d,static_cast<firstpass::BlendMode>(mode));
}
template<class T>HD T grad_d(T s,T d,int mode){
    if constexpr(std::is_same_v<T,float>)return firstpass32::blend_grad_d(s,d,static_cast<firstpass32::BlendMode>(mode));
    else return firstpass::blend_grad_d(s,d,static_cast<firstpass::BlendMode>(mode));
}
struct GaussianPrior {
    template<class T>HD static T evaluate(const Layer& layer,const T* color,T* gradient){
        T delta[3],v[3]{},cost=0;for(int r=0;r<3;++r)delta[r]=color[r]-T(layer.mean[r]);
        for(int r=0;r<3;++r){for(int s=0;s<3;++s)v[r]+=T(layer.inverse[3*r+s])*delta[s];cost+=delta[r]*v[r];}
        // Valid covariance is symmetric. This explicitly differentiates the
        // quadratic also for roundoff-asymmetric serialized matrices.
        if(gradient)for(int r=0;r<3;++r){gradient[r]=v[r];for(int s=0;s<3;++s)gradient[r]+=T(layer.inverse[3*s+r])*delta[s];}
        return cost;
    }
};
// An executable example of the device-model extension contract, not a claim
// that arbitrary host virtual functions can be launched on the GPU.
struct QuarticPrior {
    template<class T>HD static T evaluate(const Layer& layer,const T* color,T* gradient){
        T value=GaussianPrior::evaluate(layer,color,gradient);
        for(int r=0;r<3;++r){T d=color[r]-T(layer.mean[r]),d2=d*d;value+=T(layer.quartic)*d2*d2;if(gradient)gradient[r]+=T(4*layer.quartic)*d2*d;}
        return value;
    }
};

template<class T,class Prior=GaussianPrior,class Model=Layer>struct Program {
    HD static T energy(const Model* model,const Problem& problem,View<const T> x){
        int L=problem.layers;T result=0,asum=0,a2sum=0;
        for(int l=0;l<L;++l){T color[3];for(int r=0;r<3;++r)color[r]=x[L+3*l+r];result+=x[l]*Prior::evaluate(model[l],color,static_cast<T*>(nullptr));asum+=x[l];a2sum+=x[l]*x[l];}
        if(problem.minimum)result+=T(.01)*asum;
        if(problem.sparsity)result+=T(problem.sparsity_weight)*(asum/a2sum-1);
        return result;
    }
    HD static void forward(const Model* model,const Problem& problem,View<const T> x,View<T> prefix){
        int L=problem.layers;for(int r=0;r<3;++r)prefix[r]=x[L+r];prefix[3]=x[0];
        for(int l=1;l<L;++l){const auto& m=model[l];T a=x[l],d=prefix[4*(l-1)+3];
            T A=T(m.X)*a*d+T(m.Y)*a*(1-d)+T(m.Z)*d*(1-a);prefix[4*l+3]=A;
            for(int r=0;r<3;++r){T s=x[L+3*l+r],c=prefix[4*(l-1)+r];T pre=blend(s,c,m.mode)*a*d+T(m.Y)*a*(1-d)*s+T(m.Z)*d*(1-a)*c;prefix[4*l+r]=A>T(1e-12)?pre/A:pre;}
        }
    }
    HD static T evaluate(const Model* model,const Problem& problem,View<const T> x,const T* target,
        View<const T> alpha,View<const T> lambda,T rho,View<T> prefix,View<T> residual,View<T> gradient,bool derivatives=true,T* original_energy=nullptr,bool gray_constraints=true){
        int L=problem.layers;forward(model,problem,x,prefix);T energy=0,asum=0,a2sum=0;
        for(int l=0;l<L;++l){T color[3],g[3];for(int r=0;r<3;++r)color[r]=x[L+3*l+r];T D=Prior::evaluate(model[l],color,derivatives?g:nullptr);energy+=x[l]*D;
            if(derivatives){gradient[l]=D;for(int r=0;r<3;++r)gradient[L+3*l+r]=x[l]*g[r];}
            asum+=x[l];a2sum+=x[l]*x[l];}
        if(problem.minimum){energy+=T(.01)*asum;if(derivatives)for(int l=0;l<L;++l)gradient[l]+=T(.01);}
        if(problem.sparsity){energy+=T(problem.sparsity_weight)*(asum/a2sum-1);if(derivatives)for(int l=0;l<L;++l)gradient[l]+=T(problem.sparsity_weight)*(a2sum-2*x[l]*asum)/(a2sum*a2sum);}
        if(original_energy)*original_energy=energy;
        for(int r=0;r<3;++r)residual[r]=prefix[4*(L-1)+r]-target[r];
        if(problem.refinement)for(int l=0;l<L;++l)residual[3+l]=x[l]-alpha[l];
        else residual[3]=prefix[4*(L-1)+3]-1;
        int offset=3+(problem.refinement?L:1);
        for(int l=0;l<L;++l)if(gray_constraints&&model[l].gray){T norm=0;for(int r=0;r<3;++r)norm+=x[L+3*l+r]*x[L+3*l+r];norm=configured::root(norm);for(int r=0;r<3;++r)residual[offset++]=T(1.7320508075688772935)*x[L+3*l+r]-norm;}
        int C=offset;for(int r=0;r<C;++r)energy+=T(.5)*rho*residual[r]*residual[r]-lambda[r]*residual[r];
        if(!derivatives)return energy;
        T adj[4];for(int r=0;r<3;++r)adj[r]=rho*residual[r]-lambda[r];adj[3]=problem.refinement?T(0):rho*residual[3]-lambda[3];
        if(problem.refinement)for(int l=0;l<L;++l)gradient[l]+=rho*residual[3+l]-lambda[3+l];
        offset=3+(problem.refinement?L:1);
        for(int l=0;l<L;++l)if(gray_constraints&&model[l].gray){T norm=0,sum=0,v[3];for(int r=0;r<3;++r){norm+=x[L+3*l+r]*x[L+3*l+r];v[r]=rho*residual[offset]-lambda[offset];sum+=v[r];++offset;}norm=configured::root(norm);
            for(int r=0;r<3;++r)gradient[L+3*l+r]+=T(1.7320508075688772935)*v[r]-(norm>0?x[L+3*l+r]/norm*sum:T(0));}
        for(int l=L-1;l>0;--l){const auto& m=model[l];T a=x[l],d=prefix[4*(l-1)+3],A=prefix[4*l+3];
            T da=T(m.X)*d+T(m.Y)*(1-d)-T(m.Z)*d,dd=T(m.X)*a-T(m.Y)*a+T(m.Z)*(1-a);
            T ga=adj[3]*da,previous_alpha=adj[3]*dd;
            for(int r=0;r<3;++r){T s=x[L+3*l+r],c=prefix[4*(l-1)+r],f=blend(s,c,m.mode),v=adj[r],out=prefix[4*l+r];
                T ds=a*d*grad_s(s,c,m.mode)+T(m.Y)*a*(1-d);
                T dc=a*d*grad_d(s,c,m.mode)+T(m.Z)*d*(1-a);
                T pa=f*d+T(m.Y)*(1-d)*s-T(m.Z)*d*c;
                T pd=f*a-T(m.Y)*a*s+T(m.Z)*(1-a)*c;
                ga+=v*(A>T(1e-12)?(pa-out*da)/A:pa);previous_alpha+=v*(A>T(1e-12)?(pd-out*dd)/A:pd);
                gradient[L+3*l+r]+=v*(A>T(1e-12)?ds/A:ds);adj[r]=v*(A>T(1e-12)?dc/A:dc);
            }
            gradient[l]+=ga;adj[3]=previous_alpha;
        }
        gradient[0]+=adj[3];for(int r=0;r<3;++r)gradient[L+r]+=adj[r];return energy;
    }
    HD static void bounds(const Problem& p,View<const T> alpha,const T* bg,View<T> lower,View<T> upper){
        int L=p.layers;for(int k=0;k<4*L;++k){lower[k]=0;upper[k]=1;}
        if(p.refinement&&p.fixed_alpha)for(int l=0;l<L;++l)lower[l]=upper[l]=alpha[l];
        if(p.opaque)lower[0]=upper[0]=1;
        if(p.refinement&&p.smooth)for(int r=0;r<3;++r)lower[L+r]=upper[L+r]=bg[r];
    }
};
}
