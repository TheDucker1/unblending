#pragma once
#include "inverse_compositing_solver.cuh"
#include <type_traits>

// Static topology sizes, runtime blend operations and model parameters. All
// helpers inline: a point must not escape through a pointer-based callback.
#ifdef __CUDACC__
#define RF_INLINE __host__ __device__ __forceinline__
#elif defined(_MSC_VER)
#define RF_INLINE __forceinline
#else
#define RF_INLINE inline __attribute__((always_inline))
#endif
#ifdef _MSC_VER
#define RF_LAMBDA_INLINE
#define RF_NOINLINE __declspec(noinline)
#else
#define RF_LAMBDA_INLINE __attribute__((always_inline))
#define RF_NOINLINE __attribute__((noinline))
#endif
namespace register_first {
template<int I>struct Index {HD constexpr operator int()const{return I;}};
template<int Begin,int End,class F> RF_INLINE void each(F&& f) {
    if constexpr(Begin<End){f(Index<Begin>{});each<Begin+1,End>(f);}
}
template<class T> struct Model { T mean[3],precision[9],covariance[6];int mode; };
template<class T> RF_INLINE Model<T> cache(const custom_backend::Model& m) {
    Model<T> out;out.mode=m.mode;
    each<0,3>([&](auto r) RF_LAMBDA_INLINE{out.mean[r]=T(m.mean[r]);});
    each<0,9>([&](auto k) RF_LAMBDA_INLINE{out.precision[k]=T(m.inverse[k]);});
    constexpr int index[6]={0,1,2,4,5,8};
    each<0,6>([&](auto k) RF_LAMBDA_INLINE{out.covariance[k]=T(m.covariance[index[k]]);});return out;
}
template<class T,int L> struct Point {T a[L],c[L][3];};
template<class T,int L> RF_INLINE void copy(const Point<T,L>& x,Point<T,L>& y) {
    each<0,L>([&](auto l) RF_LAMBDA_INLINE{y.a[l]=x.a[l];each<0,3>([&](auto r) RF_LAMBDA_INLINE{y.c[l][r]=x.c[l][r];});});
}
template<bool Gradient,class T> RF_INLINE T gaussian(const Model<T>& m,const T* color,T* gradient=nullptr) {
    T delta[3],v[3]{},cost=0;
    each<0,3>([&](auto r) RF_LAMBDA_INLINE{delta[r]=color[r]-m.mean[r];});
    each<0,3>([&](auto r) RF_LAMBDA_INLINE{each<0,3>([&](auto s) RF_LAMBDA_INLINE{v[r]+=m.precision[3*r+s]*delta[s];});cost+=delta[r]*v[r];});
    if constexpr(Gradient)each<0,3>([&](auto r) RF_LAMBDA_INLINE{gradient[r]=v[r];each<0,3>([&](auto s) RF_LAMBDA_INLINE{gradient[r]+=m.precision[3*s+r]*delta[s];});});
    return cost;
}
template<class T,int L> RF_INLINE T energy(const Model<T>* model,general::Problem p,const Point<T,L>& point) {
    T result=0,sum=0,square=0;each<0,L>([&](auto l) RF_LAMBDA_INLINE{result+=point.a[l]*gaussian<false>(model[l],point.c[l]);sum+=point.a[l];square+=point.a[l]*point.a[l];});
    if(p.minimum)result+=T(.01)*sum;
    if(p.sparsity)result+=T(p.sparsity_weight)*(sum/square-1);return result;
}

template<class T>struct Channel {T color,destination;int kind;};
// Share substantial inverse code, but pass no point/prefix array addresses.
// The POD result lets the storage-owning caller retain its arrays in registers.
template<class T>HD RF_NOINLINE Channel<T> inverse_channel(T source,T destination,T desired,T u,int mode,int layer){
    T v=0,w=1-u,color;T preferred=configured::isfinite(destination)?configured::clip(destination):T(.5);bool conditioned=preferred!=destination;
    if(!conditioned){auto inverse=range_union::sources(range_refinement::Nearest<T,2>{source},mode,destination,desired,u,v,w);if(inverse.nearest(source,color))return {configured::clip(color),destination,0};}
    range_union::InteriorNearest<T> nearest(preferred,mode,layer);range_union::destinations(nearest,mode,T(0),T(1),desired,u,v,w);
    if(!nearest.nearest(destination,destination))return {source,destination,2};auto inverse=range_union::sources(range_refinement::Nearest<T,2>{source},mode,destination,desired,u,v,w);
    if(!inverse.nearest(source,color))return {source,destination,2};return {configured::clip(color),destination,1};
}

// Same warm-prefix / source-first / nearest-destination feasible decoder as
// reachable_cube::pixel. Prefixes are RGB only; every prefix alpha is one.
template<class T,int L> RF_INLINE int decode(const Model<T>* model,const Point<T,L>& warm,const T* target,const T* bg,Point<T,L>& out,T tolerance) {
    bool valid=true;each<0,L>([&](auto l) RF_LAMBDA_INLINE{T a=warm.a[l];valid=valid&&configured::isfinite(a)&&a>=0&&a<=1&&(l!=0||a==1);out.a[l]=a;each<0,3>([&](auto r) RF_LAMBDA_INLINE{out.c[l][r]=configured::clip(warm.c[l][r]);});});
    if(!valid)return 4;T prefix[L][3],current[3];
    each<0,3>([&](auto r) RF_LAMBDA_INLINE{valid=valid&&configured::isfinite(bg[r])&&bg[r]>=0&&bg[r]<=1;current[r]=configured::clip(warm.c[0][r]);prefix[0][r]=current[r];});if(!valid)return 4;
    each<1,L>([&](auto l) RF_LAMBDA_INLINE{each<0,3>([&](auto r) RF_LAMBDA_INLINE{current[r]=reachable_cube::over(model[l],warm.a[l],configured::clip(warm.c[l][r]),current[r]);prefix[l][r]=current[r];});});
    T goal[3],desired[3];bool projected=false;each<0,3>([&](auto r) RF_LAMBDA_INLINE{valid=valid&&configured::isfinite(target[r]);goal[r]=configured::clip(target[r]);desired[r]=goal[r];projected|=goal[r]!=target[r];});if(!valid)return 4;
    unsigned active=7;int code=0;
    each<0,L-1>([&](auto rev) RF_LAMBDA_INLINE{constexpr int l=L-1-rev;
        if(!active||code)return;T u=warm.a[l];
        each<0,3>([&](auto r) RF_LAMBDA_INLINE{if(code||!(active&(1u<<r)))return;
            auto result=inverse_channel(configured::clip(warm.c[l][r]),prefix[l-1][r],desired[r],u,model[l].mode,l);
            if(result.kind==2){code=2;return;}out.c[l][r]=result.color;
            if(result.kind==0)active&=~(1u<<r);else desired[r]=result.destination;
        });
    });if(code)return code;
    each<0,3>([&](auto r) RF_LAMBDA_INLINE{if(active&(1u<<r))out.c[0][r]=configured::clip(desired[r]);current[r]=out.c[0][r];});
    each<1,L>([&](auto l) RF_LAMBDA_INLINE{each<0,3>([&](auto r) RF_LAMBDA_INLINE{current[r]=reachable_cube::over(model[l],out.a[l],out.c[l][r],current[r]);});});
    T error=0;each<0,3>([&](auto r) RF_LAMBDA_INLINE{T d=current[r]-goal[r];error+=d*d;});if(!configured::isfinite(error)||error>tolerance*tolerance)return 3;return projected?5:0;
}

template<bool Colors,class T,int L> RF_INLINE bool proposal(const Model<T>* model,general::Problem p,const Point<T,L>& point,const T* target,T (*preferred)[3],T* alpha_gradient=nullptr) {
    T prefix[L][3],jac[L][3];each<0,3>([&](auto r) RF_LAMBDA_INLINE{prefix[0][r]=point.c[0][r];});
    each<1,L>([&](auto l) RF_LAMBDA_INLINE{T a=point.a[l];each<0,3>([&](auto r) RF_LAMBDA_INLINE{T d=prefix[l-1][r];prefix[l][r]=a==0?d:a*general::blend(point.c[l][r],d,model[l].mode)+(1-a)*d;});});
    T adj[3]={1,1,1};each<0,L-1>([&](auto rev) RF_LAMBDA_INLINE{constexpr int l=L-1-rev;T a=point.a[l];each<0,3>([&](auto r) RF_LAMBDA_INLINE{if(a==0){jac[l][r]=0;return;}T s=point.c[l][r],d=prefix[l-1][r];jac[l][r]=adj[r]*a*general::grad_s(s,d,model[l].mode);adj[r]*=a*general::grad_d(s,d,model[l].mode)+(1-a);});});each<0,3>([&](auto r) RF_LAMBDA_INLINE{jac[0][r]=adj[r];});
    T rhs[3],K[6]{},dual[3];each<0,3>([&](auto r) RF_LAMBDA_INLINE{rhs[r]=prefix[L-1][r]-target[r];});bool degenerate=false;
    constexpr int row[6]={0,0,0,1,1,2},col[6]={0,1,2,1,2,2},indices[9]={0,1,2,1,3,4,2,4,5};
    each<0,L>([&](auto l) RF_LAMBDA_INLINE{if constexpr(Colors)each<0,3>([&](auto r) RF_LAMBDA_INLINE{preferred[l][r]=point.c[l][r];});T a=point.a[l];if(a==0){if(jac[l][0]!=0||jac[l][1]!=0||jac[l][2]!=0)degenerate=true;return;}
        T gradient[3],j[3],normalized[3];gaussian<true>(model[l],point.c[l],gradient);
        each<0,3>([&](auto r) RF_LAMBDA_INLINE{j[r]=jac[l][r];T move=0;each<0,3>([&](auto s) RF_LAMBDA_INLINE{move+=T(.5)*model[l].covariance[indices[3*r+s]]*gradient[s];});if constexpr(Colors)preferred[l][r]=point.c[l][r]-move;rhs[r]-=j[r]*move;normalized[r]=j[r]/(T(2)*a);jac[l][r]=normalized[r];});
        each<0,6>([&](auto k) RF_LAMBDA_INLINE{K[k]+=j[row[k]]*model[l].covariance[k]*normalized[col[k]];});
    });T trace=K[0]+K[3]+K[5];if(degenerate||!configured::isfinite(trace)||trace<0)return false;
    if(trace>0){if(!normal_refinement::solve3(K,trace*T(sizeof(T)==4?1e-6:1e-12),rhs,dual))return false;}else each<0,3>([&](auto r) RF_LAMBDA_INLINE{dual[r]=0;});
    bool finite=true;each<0,L>([&](auto l) RF_LAMBDA_INLINE{if(point.a[l]==0)return;T gradient[3];if constexpr(!Colors)gaussian<true>(model[l],point.c[l],gradient);
        each<0,3>([&](auto r) RF_LAMBDA_INLINE{T predicted;if constexpr(Colors)predicted=preferred[l][r];else{T move=0;each<0,3>([&](auto s) RF_LAMBDA_INLINE{move+=T(.5)*model[l].covariance[indices[3*r+s]]*gradient[s];});predicted=point.c[l][r]-move;}
            each<0,3>([&](auto s) RF_LAMBDA_INLINE{predicted-=model[l].covariance[indices[3*r+s]]*jac[l][s]*dual[s];});finite=finite&&configured::isfinite(predicted);if constexpr(Colors)preferred[l][r]=predicted;
        });
    });if(!finite)return false;
    if constexpr(!Colors){T sum=0,square=0;each<0,L>([&](auto l) RF_LAMBDA_INLINE{alpha_gradient[l]=gaussian<false>(model[l],point.c[l])+(p.minimum?T(.01):T(0));sum+=point.a[l];square+=point.a[l]*point.a[l];});
        if(p.sparsity)each<0,L>([&](auto l) RF_LAMBDA_INLINE{alpha_gradient[l]+=T(p.sparsity_weight)*(square-T(2)*point.a[l]*sum)/(square*square);});
        each<0,3>([&](auto r) RF_LAMBDA_INLINE{adj[r]=dual[r];});each<0,L-1>([&](auto rev) RF_LAMBDA_INLINE{constexpr int l=L-1-rev;T a=point.a[l],ga=0;each<0,3>([&](auto r) RF_LAMBDA_INLINE{T s=point.c[l][r],d=prefix[l-1][r],v=adj[r];ga+=v*(general::blend(s,d,model[l].mode)-d);adj[r]=v*(a*general::grad_d(s,d,model[l].mode)+(1-a));});alpha_gradient[l]+=ga;});
    }return true;
}

template<class T,int L> RF_INLINE int improve(const Model<T>* model,general::Problem p,Point<T,L>& point,const T* target,const T* bg,T tolerance,int rounds,int& accepted,int& values) {
    accepted=values=0;int status=decode(model,point,target,bg,point,tolerance);if(status!=0&&status!=5)return status;T best=energy(model,p,point);++values;if(!configured::isfinite(best))return 2;if(rounds==0)return status;
    T goal[3];each<0,3>([&](auto r) RF_LAMBDA_INLINE{goal[r]=configured::clip(target[r]);});
    for(int iteration=0;iteration<rounds;++iteration){T preferred[L][3];if(!proposal<true>(model,p,point,goal,preferred))break;bool moved=false;
        for(int attempt=0;attempt<5;++attempt){Point<T,L> trial;each<0,L>([&](auto l) RF_LAMBDA_INLINE{trial.a[l]=point.a[l];each<0,3>([&](auto r) RF_LAMBDA_INLINE{preferred[l][r]=point.c[l][r]+T(.5)*(preferred[l][r]-point.c[l][r]);trial.c[l][r]=preferred[l][r];});});
            int code=decode(model,trial,target,bg,trial,tolerance);T cost=energy(model,p,trial);++values;if((code==0||code==5)&&configured::isfinite(cost)&&cost<best){copy(trial,point);best=cost;status=code;++accepted;moved=true;break;}}
        if(!moved)break;
    }return status;
}
template<class T,int L> RF_INLINE int first(const Model<T>* model,general::Problem p,Point<T,L>& point,const T* target,const T* bg,T tolerance,int rounds,int& accepted,int& values) {
    p.smooth=false;p.fixed_alpha=false;accepted=values=0;each<0,L>([&](auto l) RF_LAMBDA_INLINE{point.a[l]=configured::clip(point.a[l]);each<0,3>([&](auto r) RF_LAMBDA_INLINE{point.c[l][r]=configured::clip(point.c[l][r]);});});point.a[0]=1;
    int a=0,v=0;int status=improve(model,p,point,target,bg,tolerance,0,a,v);values=v;if(status!=0&&status!=5)return status;T best=energy(model,p,point);++values;
    for(int outer=0;outer<rounds;++outer){bool color_moved=false;
        {Point<T,L> candidate;copy(point,candidate);int code=improve(model,p,candidate,target,bg,tolerance,3,a,v);values+=v;T cost=energy(model,p,candidate);++values;if((code==0||code==5)&&configured::isfinite(cost)&&cost<best){copy(candidate,point);best=cost;accepted+=a;status=code;color_moved=true;}}
        T gradient[L];if(!proposal<false>(model,p,point,target,static_cast<T(*)[3]>(nullptr),gradient))break;++values;T scale=1;each<1,L>([&](auto l) RF_LAMBDA_INLINE{T g=std::abs(gradient[l]);if(g>scale)scale=g;});if(!configured::isfinite(scale))break;
        bool moved=false;T amount=T(.5);for(int attempt=0;attempt<6;++attempt){Point<T,L> trial;copy(point,trial);each<0,L>([&](auto l) RF_LAMBDA_INLINE{trial.a[l]=configured::clip(point.a[l]-amount*gradient[l]/scale);});trial.a[0]=1;
            int code=improve(model,p,trial,target,bg,tolerance,0,a,v);values+=v;T cost=energy(model,p,trial);++values;if((code==0||code==5)&&configured::isfinite(cost)&&cost<best){copy(trial,point);best=cost;status=code;++accepted;moved=true;break;}amount*=T(.5);}
        if(!moved&&!color_moved)break;
    }return status;
}
}
#undef RF_INLINE
#undef RF_LAMBDA_INLINE
#undef RF_NOINLINE
