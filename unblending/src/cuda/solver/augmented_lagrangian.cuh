#pragma once
#include "model/pixel_problem.cuh"
#include <limits>

namespace general {
template<class T>HD T minimum_value(T a,T b){return a<b?a:b;}
template<class T>HD T maximum_value(T a,T b){return a>b?a:b;}
struct Options {
    int history=6,inner=256,outer=22,starts=1;
    bool canonical=false,correction=false,project_gray=false;
    double inner_tolerance=.005,outer_tolerance=.005;
};
struct Work {unsigned values=0,gradients=0,iterations=0,backtracks=0,outer=0,correction_values=0,corrections=0;};

// A general compatibility baseline: bounded projected L-BFGS/AL. Histories,
// multipliers and all trial evaluations remain GPU-side. State is coalesced
// across pixel workers, with a small history rather than one64-pair register
// allocation per thread. This is not claimed equivalent to NLopt's active-set
// history algebra or selected for speed before actual qualification.
template<class T,class Prior,class Model=Layer>struct Optimizer {
    using P=Program<T,Prior,Model>;
    HD static void project_gray(const Model* model,const Problem& p,View<T> x){
        int L=p.layers;for(int l=0;l<L;++l)if(model[l].gray){int k=L+3*l;if(p.refinement&&p.smooth&&l==0)continue;
            if(x[k]!=x[k+1]||x[k+1]!=x[k+2]){T g=(x[k]+x[k+1]+x[k+2])/T(3);g=maximum_value(T(0),minimum_value(T(1),g));for(int r=0;r<3;++r)x[k+r]=g;}}
    }
    HD static void project_gray_gradient(const Model* model,const Problem& p,View<T> gradient){
        int L=p.layers;for(int l=0;l<L;++l)if(model[l].gray){int k=L+3*l;T g=(gradient[k]+gradient[k+1]+gradient[k+2])/T(3);for(int r=0;r<3;++r)gradient[k+r]=g;}
    }
    // Apply the inverse limited-memory BFGS model in the CURRENT free space.
    // Restricting only the final direction is wrong: every dot product,
    // curvature reciprocal and initial scale must use the same free mask.
    HD static void inverse_step(int N,int H,int valid,int next,View<const T> free,View<const T> g,
        View<const T> s,View<const T> y,View<T> q,View<T> coeff,View<T> inverse){
        for(int k=0;k<N;++k)q[k]=free[k]?g[k]:T(0);
        T scaling=1;bool have_scaling=false;
        for(int j=0;j<valid;++j){int slot=(next-1-j+H)%H;T dot=0,sy=0,ss=0,yy=0;
            for(int k=0;k<N;++k)if(free[k]){T a=s[slot*N+k],b=y[slot*N+k];dot+=a*q[k];sy+=a*b;ss+=a*a;yy+=b*b;}
            inverse[slot]=sy>T(1e-9)*configured::root(ss*yy)&&sy>0&&std::isfinite(1/sy)?T(1)/sy:T(0);
            coeff[slot]=dot*inverse[slot];
            for(int k=0;k<N;++k)if(free[k])q[k]-=coeff[slot]*y[slot*N+k];
            if(!have_scaling&&inverse[slot]>0&&yy>0){scaling=sy/yy;have_scaling=true;}
        }
        scaling=maximum_value(T(1e-12),minimum_value(T(1e3),scaling));for(int k=0;k<N;++k)q[k]*=scaling;
        for(int j=valid-1;j>=0;--j){int slot=(next-1-j+H)%H;T dot=0;
            for(int k=0;k<N;++k)if(free[k])dot+=y[slot*N+k]*q[k];T a=coeff[slot]-dot*inverse[slot];
            for(int k=0;k<N;++k)if(free[k])q[k]+=a*s[slot*N+k];
        }
    }
    HD static T bound(const Problem& p,View<const T> alpha,const T* bg,int k,bool high){
        int L=p.layers;if(k<L){if(p.opaque&&k==0)return 1;if(p.refinement&&p.fixed_alpha)return alpha[k];}
        if(p.refinement&&p.smooth&&k>=L&&k<L+3)return bg[k-L];return high?T(1):T(0);
    }
    HD static int workspace(const Problem& p,int C,const Options& opt){return (8+2*opt.history)*4*p.layers+2*opt.history+2*C;}
    HD static bool canonicalize(const Model* model,const Problem& p,View<const T> alpha,const T* bg,View<T> x){
        bool changed=false;int L=p.layers;
        // At exactly zero source alpha, its color contributes neither to the
        // composition nor to a_l*D(c_l). Choose the representative color on
        // this flat set so a stale invisible color does not block activation.
        // Gray constraints depend on color even at zero alpha: never reset them.
        for(int l=1;l<L;++l)if(x[l]==0&&!model[l].gray)for(int r=0;r<3;++r){int k=L+3*l+r;T v=T(model[l].mean[r]);
            v=maximum_value(bound(p,alpha,bg,k,false),minimum_value(bound(p,alpha,bg,k,true),v));
            changed|=x[k]!=v;x[k]=v;}
        return changed;
    }
    HD static void solve(const Model* model,const Problem& p,const T* target,View<const T> alpha,const T* bg,
        View<const T> initial,View<T> output,View<T> memory,const Options& opt,Work& work){
        int L=p.layers,N=4*L,H=opt.history;bool eliminate_gray=opt.project_gray;
        // Fixed non-gray background data cannot satisfy an exact gray color.
        // Retain the original equations for that inconsistent/tolerance case.
        if(p.refinement&&p.smooth&&model[0].gray&&(bg[0]!=bg[1]||bg[1]!=bg[2]))eliminate_gray=false;
        int C=eliminate_gray?3+(p.refinement?L:1):p.constraints(model);
        auto at=[&](int offset){return View<T>{memory.data+size_t(offset)*memory.stride,memory.stride};};
        View<T> x=at(0),g=at(N),direction=at(2*N),trial=at(3*N),newg=at(4*N),outer_x=at(5*N),best=at(6*N),prefix=at(7*N);
        View<T> history_s=at(8*N),history_y=at((8+H)*N),coeff=at((8+2*H)*N),inverse=at((8+2*H)*N+H);
        View<T> lambda=at((8+2*H)*N+2*H),residual=at((8+2*H)*N+2*H+C);
        auto constant=[](View<T> v){return View<const T>{v.data,v.stride};};
        T best_norm=INFINITY,best_energy=INFINITY;bool have_best=false;
        auto evaluate=[&](View<T> point,View<T> gradient,T rho,bool derivative){++work.values;if(derivative)++work.gradients;
            if(eliminate_gray)project_gray(model,p,point);
            T energy;T value=P::evaluate(model,p,constant(point),target,alpha,constant(lambda),rho,prefix,residual,gradient,derivative,&energy,!eliminate_gray);
            if(derivative&&eliminate_gray)project_gray_gradient(model,p,gradient);
            T norm=0,tol=T(opt.outer_tolerance*opt.outer_tolerance);for(int r=0;r<C;++r)norm+=residual[r]*residual[r];
            bool feasible=norm<tol;
            if(std::isfinite(energy)&&std::isfinite(norm)&&(!have_best||(feasible&&(best_norm>=tol||energy<best_energy))||(!feasible&&best_norm>=tol&&norm<best_norm))){
                have_best=true;best_norm=norm;best_energy=energy;for(int k=0;k<N;++k)best[k]=point[k];}
            return value;};
        for(int start=0;start<opt.starts;++start){
            for(int k=0;k<N;++k){T value=initial[k];
                if(!p.refinement&&start==1){value=k<L?(k==0?T(1):T(0)):(k<L+3?configured::clip(target[k-L]):T(model[(k-L)/3].mean[(k-L)%3]));}
                if(p.refinement&&start&&k>=L)value=T(model[(k-L)/3].mean[(k-L)%3]);
                if(start==2){if(k<L)value=p.refinement?initial[k]:T(.5);else value=maximum_value(T(.05),minimum_value(T(.95),T(model[(k-L)/3].mean[(k-L)%3])));}
                T lower=bound(p,alpha,bg,k,false),upper=bound(p,alpha,bg,k,true);
                x[k]=value<lower?lower:(value>upper?upper:value);}
            if(start==2){
                // Dodge/Burn's clipped branches erase source derivatives.
                // Build an additional seed inside the unsaturated branch when
                // the current destination permits it. Keep original/feasible
                // starts too; no monotonicity assumption or target change.
                T dest[3];for(int r=0;r<3;++r)dest[r]=x[L+r];T da=x[0];
                for(int l=1;l<L;++l){const auto& m=model[l];T a=x[l],A=T(m.X)*a*da+T(m.Y)*a*(1-da)+T(m.Z)*da*(1-a);
                    for(int r=0;r<3;++r){int k=L+3*l+r;T s=x[k],d=dest[r];
                        if(d>=0&&d<=1&&m.mode==6)s=minimum_value(s,maximum_value(T(0),1-d-T(.05)));
                        if(d>=0&&d<=1&&m.mode==7)s=maximum_value(s,minimum_value(T(1),1-d+T(.05)));
                        s=maximum_value(bound(p,alpha,bg,k,false),minimum_value(bound(p,alpha,bg,k,true),s));x[k]=s;
                        T pre=blend(s,d,m.mode)*a*da+T(m.Y)*a*(1-da)*s+T(m.Z)*da*(1-a)*d;dest[r]=A>T(1e-12)?pre/A:pre;
                    }da=A;
                }
            }
            if(eliminate_gray)project_gray(model,p,x);
            for(int r=0;r<C;++r)lambda[r]=0;T rho=100;
            for(int outer=0;outer<opt.outer;++outer){++work.outer;if(opt.canonical)canonicalize(model,p,alpha,bg,x);for(int k=0;k<N;++k)outer_x[k]=x[k];
                T f=evaluate(x,g,rho,true),oldnorm=0;for(int r=0;r<C;++r)oldnorm+=residual[r]*residual[r];int valid=0,next=0;
                for(int iteration=0;iteration<opt.inner;++iteration){++work.iterations;T pg2=0;
                    for(int k=0;k<N;++k){T lo=bound(p,alpha,bg,k,false),hi=bound(p,alpha,bg,k,true);bool fixed=lo==hi||(x[k]<=lo&&g[k]>0)||(x[k]>=hi&&g[k]<0);trial[k]=fixed?0:1;direction[k]=fixed?0:g[k];pg2+=direction[k]*direction[k];}
                    if(pg2<T(1e-16)||!std::isfinite(f))break;
                    inverse_step(N,H,valid,next,constant(trial),constant(g),constant(history_s),constant(history_y),direction,coeff,inverse);
                    T descent=0;
                    for(int k=0;k<N;++k){T lo=bound(p,alpha,bg,k,false),hi=bound(p,alpha,bg,k,true),d=-direction[k];if(lo==hi||(x[k]<=lo&&d<0)||(x[k]>=hi&&d>0))d=0;direction[k]=d;descent+=d*g[k];}
                    if(!(descent<T(0))){valid=0;next=0;for(int k=0;k<N;++k){T lo=bound(p,alpha,bg,k,false),hi=bound(p,alpha,bg,k,true);direction[k]=(lo==hi||(x[k]<=lo&&g[k]>0)||(x[k]>=hi&&g[k]<0))?0:-g[k];}}
                    bool accepted=false;T candidate=f;
                    for(int attempt=0;attempt<2&&!accepted;++attempt){T step=1;
                        for(int backtrack=0;backtrack<32;++backtrack){T slope=0,motion=0;
                            for(int k=0;k<N;++k){T lo=bound(p,alpha,bg,k,false),hi=bound(p,alpha,bg,k,true);trial[k]=maximum_value(lo,minimum_value(hi,x[k]+step*direction[k]));T delta=trial[k]-x[k];slope+=g[k]*delta;motion+=delta*delta;}
                            if(motion==0)break;
                            candidate=evaluate(trial,newg,rho,false);++work.backtracks;
                            if(slope<0&&std::isfinite(candidate)&&candidate<=f+T(.0001)*slope){accepted=true;break;}step*=T(.5);
                        }
                        if(!accepted){valid=0;next=0;for(int k=0;k<N;++k){T lo=bound(p,alpha,bg,k,false),hi=bound(p,alpha,bg,k,true);direction[k]=(lo==hi||(x[k]<=lo&&g[k]>0)||(x[k]>=hi&&g[k]<0))?0:-g[k];}}
                    }
                    if(!accepted)break;
                    bool canonical=opt.canonical&&canonicalize(model,p,alpha,bg,trial);
                    candidate=evaluate(trial,newg,rho,true);T sy=0,ss=0,yy=0,maxmotion=0;
                    for(int k=0;k<N;++k){T s=trial[k]-x[k],y=newg[k]-g[k];sy+=s*y;ss+=s*s;yy+=y*y;maxmotion=maximum_value(maxmotion,std::abs(s)/(T(1)+std::abs(x[k])));}
                    if(canonical){valid=0;next=0;}
                    else if(sy>T(1e-9)*configured::root(ss*yy)&&sy>0&&std::isfinite(1/sy)){
                        for(int k=0;k<N;++k){history_s[next*N+k]=trial[k]-x[k];history_y[next*N+k]=newg[k]-g[k];}inverse[next]=1/sy;next=(next+1)%H;valid=minimum_value(valid+1,H);}
                    T delta=std::abs(f-candidate),size=T(.5)*(std::abs(f)+std::abs(candidate));for(int k=0;k<N;++k){x[k]=trial[k];g[k]=newg[k];}f=candidate;
                    if(!canonical&&(delta<T(opt.inner_tolerance)*size||maxmotion<T(opt.inner_tolerance)))break;
                }
                f=evaluate(x,g,rho,true);T norm=0,motion=0;
                for(int r=0;r<C;++r){T v=residual[r];norm+=v*v;}
                bool feasible=norm<T(opt.outer_tolerance*opt.outer_tolerance);
                for(int r=0;r<C;++r)lambda[r]-=rho*residual[r];if(norm>T(.0625)*oldnorm)rho*=10;
                for(int k=0;k<N;++k){T d=x[k]-outer_x[k];motion+=d*d;}
                if(motion<T(opt.outer_tolerance*opt.outer_tolerance)&&feasible)break;
            }
        }
        for(int k=0;k<N;++k)output[k]=have_best?best[k]:initial[k];
    }
};
}
