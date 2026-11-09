#pragma once
#include "model/color_model.hpp"
#include "solver/feasibility/constraint_restoration.cuh"
#include "solver/three_layer/constraint_projection.cuh"
#include "solver/feasibility/algebraic_projection.cuh"
#include "solver/three_layer/trial_step.cuh"
#include "solver/three_layer/gray_warm_start.cuh"

namespace custom_refinement {
template<class T>HD T local_metric(const custom_backend::Model&,const T*,T*,T*,T* = nullptr);
}

#include "solver/three_layer/search_direction.cuh"

namespace coupled_sqp {
#if defined(__CUDA_ARCH__)
constexpr bool host_reduced_math=false;
#else
constexpr bool host_reduced_math=true;
#endif
using general::View;
using Model=custom_backend::Model;
using Prior=general::PolynomialPrior;
#ifdef __CUDACC__
#define COUPLED_OUTLINED __noinline__
#else
#define COUPLED_OUTLINED
#endif

// Metric SQP with at most four multipliers. No history or layer-sized local
// arrays: all O(layers) state is coalesced across independent pixel workers.
template<class T>struct Solver {
    using P=general::Program<T,Prior,Model>;
    using R=general::ConstraintRestore<T,Model>;
    int L,N,rows;
    const Model* model;
    general::Problem p;
    View<T> memory;

    HD View<T> at(int offset)const{return {memory.data+size_t(offset)*memory.stride,memory.stride};}
    HD static View<const T> cv(View<T> x){return {x.data,x.stride};}
    HD View<T> metric()const{return at(16*N);}
    HD View<T> alpha_metric()const{return at(16*N+6*L);}

    HD bool gradient_metric(View<const T> point){probe::hit(probe::Metric);
        auto g=at(11*N),B=metric(),aB=alpha_metric();T sum=0,square=0;
        for(int l=0;l<L;++l){
            T c[3],gradient[3],covariance[6],gray;
            for(int r=0;r<3;++r)c[r]=point[L+3*l+r];
            T distance=custom_refinement::local_metric(model[l],c,gradient,covariance,&gray);
            T alpha=point[l],scale=alpha>T(.05)?alpha:T(.05);
            g[l]=p.refinement?T(0):distance;aB[l]=p.refinement?T(0):T(1)/(1+std::abs(distance));
            for(int r=0;r<3;++r)g[L+3*l+r]=alpha*gradient[r];
            bool valid=true;
            for(int k=0;k<6;++k){B[6*l+k]=model[l].gray?gray/scale:T(.5)*covariance[k]/scale;valid&=configured::isfinite(B[6*l+k]);}
            // Numerical fallback is a positive isotropic metric, restricted to
            // the gray direction if requested; the actual prior is unchanged.
            if(!valid){for(int k=0;k<6;++k)B[6*l+k]=model[l].gray?T(1)/T(3):T(0);if(!model[l].gray)B[6*l]=B[6*l+3]=B[6*l+5]=1;}
            sum+=alpha;square+=alpha*alpha;
        }
        if(p.minimum)for(int l=0;l<L;++l)g[l]+=T(.01);
        if(p.sparsity){if(!(square>0))return false;for(int l=0;l<L;++l)g[l]+=T(p.sparsity_weight)*(square-2*point[l]*sum)/(square*square);}
        for(int k=0;k<N;++k)if(!configured::isfinite(g[k]))return false;
        return true;
    }

    HD bool direction(View<const T> point,const T* target){
        auto prefix=at(0),J=at(12*N),free=at(18*N),g=at(11*N),step=at(10*N),B=metric(),aB=alpha_metric();
        P::forward(model,p,point,prefix);sqp_math::jacobian(model,p,point,cv(prefix),J);
        for(int k=0;k<N;++k)free[k]=!(k<L?(p.refinement||(p.opaque&&k==0)):(p.refinement&&p.smooth&&k<L+3));
        T residual[4];for(int r=0;r<3;++r)residual[r]=prefix[4*(L-1)+r]-target[r];residual[3]=prefix[4*(L-1)+3]-1;
        const int index[9]={0,1,2,1,3,4,2,4,5};
        #pragma unroll 1
        for(int active_pass=0;active_pass<4;++active_pass){probe::hit(probe::ActivePass);
            T gram[16]{},rhs[4],dual[4]{};for(int r=0;r<rows;++r)rhs[r]=residual[r];
            for(int l=0;l<L;++l){
                T j[3],bg[3]{};
                for(int r=0;r<3;++r){j[r]=J[r*N+L+3*l+r];for(int s=0;s<3;++s)bg[r]+=B[6*l+index[3*r+s]]*free[L+3*l+r]*free[L+3*l+s]*g[L+3*l+s];}
                for(int r=0;r<3;++r){rhs[r]-=j[r]*bg[r];for(int s=0;s<3;++s)gram[4*r+s]+=j[r]*B[6*l+index[3*r+s]]*free[L+3*l+r]*free[L+3*l+s]*j[s];}
                T alphaB=aB[l]*free[l];
                if(alphaB!=0)for(int r=0;r<rows;++r){rhs[r]-=J[r*N+l]*alphaB*g[l];for(int s=0;s<rows;++s)gram[4*r+s]+=J[r*N+l]*alphaB*J[s*N+l];}
            }
            // With no active image direction, a pure prior step may still
            // improve invisible colors. A zero dual handles that flat case.
            T trace=0;for(int r=0;r<rows;++r)trace+=gram[4*r+r];
            if(trace>0&&!R::solve(rows,gram,rhs,dual))return false;
            if(!configured::isfinite(trace)||trace<0)return false;
            for(int l=0;l<L;++l){
                T value=g[l];for(int r=0;r<rows;++r)value+=J[r*N+l]*dual[r];step[l]=-aB[l]*free[l]*value;
                T color[3];for(int r=0;r<3;++r)color[r]=g[L+3*l+r]+J[r*N+L+3*l+r]*dual[r];
                for(int r=0;r<3;++r){T v=0;for(int s=0;s<3;++s)v-=B[6*l+index[3*r+s]]*free[L+3*l+r]*free[L+3*l+s]*color[s];step[L+3*l+r]=v;}
            }
            bool changed=false;
            for(int k=0;k<N;++k)if(free[k]&&((point[k]<=0&&step[k]<0)||(point[k]>=1&&step[k]>0))){free[k]=0;changed=true;}
            if(!changed)return true;
        }
        // Bounded nonlinear restoration decides feasibility after this limited
        // active-set approximation; an accepted output cannot bypass it.
        for(int k=0;k<N;++k)if((point[k]<=0&&step[k]<0)||(point[k]>=1&&step[k]>0))step[k]=0;
        return true;
    }

    HD static COUPLED_OUTLINED int run(const Model* model,general::Problem p,View<const T> initial,
        View<const T> target,View<const T> alpha,View<const T> background,View<T> point,View<T> memory,
        T tolerance,int rounds,int& accepted,int& values){
        Solver s{p.layers,4*p.layers,sqp_math::rows(model,p),model,p,memory};accepted=values=0;
        T wanted[3],bg[3];for(int r=0;r<3;++r){wanted[r]=target[r];bg[r]=background[r];}
        if(p.refinement&&p.smooth&&model[0].gray&&(bg[0]!=bg[1]||bg[1]!=bg[2]))return 8;
        for(int k=0;k<s.N;++k)point[k]=initial[k];R::project(model,p,alpha,bg,point);
        T norm=R::norm(model,p,wanted,alpha,cv(point));
        if(p.refinement&&!(norm<=tolerance*tolerance*T(.0025))&&algebraic_retract::Inverse<T>::eligible(model,p,cv(point))){
            auto trial=s.at(9*s.N);for(int k=0;k<s.N;++k)trial[k]=point[k];
            bool direct=gray_warm_seed::run(model,p,wanted,cv(point),trial);
            if(!direct){P::forward(model,p,cv(point),s.at(0));general::rgba_jacobian(model,p,cv(point),cv(s.at(0)),s.at(12*s.N),true);
                direct=algebraic_retract::Inverse<T>::run(model,p,wanted,trial,s.at(0),cv(s.at(12*s.N)));}
            if(!direct&&!p.refinement){general::Problem seed_problem=p;seed_problem.refinement=true;
                direct=R::gray_three_seed(model,seed_problem,wanted,cv(point),bg,trial);}
            if(direct){
                T trial_norm=R::norm(model,p,wanted,alpha,cv(trial));
                if(trial_norm<norm){for(int k=0;k<s.N;++k)point[k]=trial[k];norm=trial_norm;}
            }
        }
        if(!(norm<=tolerance*tolerance*T(.0025))){probe::hit(probe::InitialRestore);if constexpr(std::is_same_v<T,float> && host_reduced_math){if(register_three::eligible(model,p))register_three::scalar_initial(model,p,wanted,point,tolerance);else R::run(model,p,wanted,alpha,bg,initial,point,memory,tolerance);}else R::run(model,p,wanted,alpha,bg,initial,point,memory,tolerance);norm=R::norm(model,p,wanted,alpha,cv(point));}
        if(!(norm<=tolerance*tolerance))return 3;
        probe::hit(probe::Energy);T best=P::energy(model,p,cv(point));++values;if(!configured::isfinite(best))return 2;
        auto candidate=s.at(9*s.N),step=s.at(10*s.N);
        #pragma unroll 1
        for(int iteration=0;iteration<rounds;++iteration){probe::hit(probe::OuterIteration);
            bool ok;if constexpr(std::is_same_v<T,float> && host_reduced_math){if(register_three::eligible(model,p)){register_three::Point delta;register_three::V anchor,t{wanted[0],wanted[1],wanted[2]};ok=register_three::proposal(model,p,register_three::load(point),t,delta,anchor,model[1].mode,model[2].mode);if(ok){auto out=s.at(10*s.N);register_three::store(out,delta);out[0]=0;auto j=s.at(12*s.N);j[6]=anchor.r;j[s.N+7]=anchor.g;j[2*s.N+8]=anchor.b;}}else ok=s.gradient_metric(cv(point))&&s.direction(cv(point),wanted);}else ok=s.gradient_metric(cv(point))&&s.direction(cv(point),wanted);if(!ok)break;
            T limit=1,magnitude=0;
            for(int k=0;k<s.N;++k){T d=step[k];if(!configured::isfinite(d))return 2;T a=std::abs(d);if(a>magnitude)magnitude=a;if(d>0){T t=(1-point[k])/d;if(t<limit)limit=t;}else if(d<0){T t=-point[k]/d;if(t<limit)limit=t;}}
            if(magnitude<=T(1e-10)||!(limit>0))break;
            if(magnitude*limit>T(.25))limit=T(.25)/magnitude;
            bool moved=false;
            #pragma unroll 1
            for(int backtrack=0;backtrack<6;++backtrack){probe::hit(probe::OuterTrial);
                T next_norm;
                if constexpr(std::is_same_v<T,float> && host_reduced_math){if(register_three::eligible(model,p)){auto c=register_three::advance(register_three::load(point),register_three::load(step),limit);auto jac=s.at(12*s.N);register_three::V anchor{jac[6],jac[s.N+7],jac[2*s.N+8]},t{wanted[0],wanted[1],wanted[2]};next_norm=register_three::trial_restore(c,t,anchor,model[1].mode,model[2].mode,tolerance);register_three::store(candidate,c);}else{                for(int k=0;k<s.N;++k)candidate[k]=point[k]+limit*step[k];R::project(model,p,alpha,bg,candidate);
                next_norm=R::norm(model,p,wanted,alpha,cv(candidate));
                if(!(next_norm<=tolerance*tolerance*T(.0025))){
                    if(algebraic_retract::Inverse<T>::eligible(model,p,cv(candidate))){
                        bool solved=algebraic_retract::Inverse<T>::run(model,p,wanted,candidate,s.at(0),cv(s.at(12*s.N)));
                        next_norm=solved?R::norm(model,p,wanted,alpha,cv(candidate)):T(INFINITY);
                    }else{probe::hit(probe::OuterRestore);R::run(model,p,wanted,alpha,bg,cv(point),candidate,memory,tolerance,8,1);next_norm=R::norm(model,p,wanted,alpha,cv(candidate));}
                }
}}else{                for(int k=0;k<s.N;++k)candidate[k]=point[k]+limit*step[k];R::project(model,p,alpha,bg,candidate);
                next_norm=R::norm(model,p,wanted,alpha,cv(candidate));
                if(!(next_norm<=tolerance*tolerance*T(.0025))){
                    if(algebraic_retract::Inverse<T>::eligible(model,p,cv(candidate))){
                        bool solved=algebraic_retract::Inverse<T>::run(model,p,wanted,candidate,s.at(0),cv(s.at(12*s.N)));
                        next_norm=solved?R::norm(model,p,wanted,alpha,cv(candidate)):T(INFINITY);
                    }else{probe::hit(probe::OuterRestore);R::run(model,p,wanted,alpha,bg,cv(point),candidate,memory,tolerance,8,1);next_norm=R::norm(model,p,wanted,alpha,cv(candidate));}
                }
}
                if(next_norm<=tolerance*tolerance){probe::hit(probe::Energy);T energy;if constexpr(std::is_same_v<T,float> && host_reduced_math){energy=register_three::eligible(model,p)?register_three::energy(model,p,register_three::load(candidate)):P::energy(model,p,cv(candidate));}else energy=P::energy(model,p,cv(candidate));++values;
                    if(configured::isfinite(energy)&&energy<best){for(int k=0;k<s.N;++k)point[k]=candidate[k];best=energy;moved=true;++accepted;probe::hit(probe::Accepted);break;}}
                limit*=T(.5);
            }
            if(!moved)break;
        }
        return 0;
    }
};
#undef COUPLED_OUTLINED
}
