#pragma once
#include "reachable_interval_unions.cuh"
#include <limits>

namespace general {
template<class T,class Model>struct ConstraintRestore;
#ifdef __CUDACC__
#define GRAY_SEED_OUTLINED __noinline__
#else
#define GRAY_SEED_OUTLINED
#endif

// For two layers with a free RGB background and a gray source s, fixed alpha
// gives t_r = u*f(s,d_r)+v*s+w*d_r. With u,w>=0, the native map over d in[0,1]
// is monotone except Difference (split at d=s); Exclusion is affine. Its range
// extrema are affine in s, with one source split for HardLight. Intersect the
// inequalities low(s)<=min(target), high(s)>=max(target), then invert d_r.
// Native residual validation handles Dodge/Burn's strict epsilon branches.
// A common-color source is also a valid INITIAL subset for an unrestricted RGB
// source. Its subsequent optimization retains three free color coordinates.
template<class T,class Model>struct TwoLayerGraySeed {
    HD static bool restrict_line(T slope,T bias,T bound,T& lo,T& hi){
        if(slope>0){T value=(bound-bias)/slope;if(value<hi)hi=value;}
        else if(slope<0){T value=(bound-bias)/slope;if(value>lo)lo=value;}
        else if(bias>bound)return false;
        return lo<=hi;
    }
    HD static T value(const Model& m,T a,T d,T A,T s,T bg){return range_union::native_color(m,a,d,A,s,bg);}
    HD static bool invert(const Model& m,T a,T d,T A,T s,T target,T preferred,T tolerance,T& answer){
        bool have=false;T best=INFINITY;int pieces=m.mode==10?2:1;
        for(int piece=0;piece<pieces;++piece){
            T lo=pieces==2&&piece==1?s:T(0),hi=pieces==2&&piece==0?s:T(1);
            T left=value(m,a,d,A,s,lo),right=value(m,a,d,A,s,hi),minimum=left<right?left:right,maximum=left>right?left:right;
            T slack=T(64)*std::numeric_limits<T>::epsilon()*(1+std::abs(target));
            if(target<minimum-slack||target>maximum+slack)continue;
            T candidate=preferred<lo?lo:(preferred>hi?hi:preferred);
            if(std::abs(value(m,a,d,A,s,candidate)-target)>tolerance/T(100)){
                bool increasing=right>=left;
                #pragma unroll 1
                for(int step=0;step<(sizeof(T)==4?24:52);++step){T mid=(lo+hi)/2,y=value(m,a,d,A,s,mid);if((y<target)==increasing)lo=mid;else hi=mid;}
                T error_lo=std::abs(value(m,a,d,A,s,lo)-target),error_hi=std::abs(value(m,a,d,A,s,hi)-target);candidate=error_lo<=error_hi?lo:hi;
            }
            if(std::abs(value(m,a,d,A,s,candidate)-target)>tolerance/T(100))continue;
            T distance=std::abs(candidate-preferred);if(!have||distance<best){answer=candidate;best=distance;have=true;}
        }
        return have;
    }
    HD static bool fixed_alpha(const Model* model,Problem p,const T* target,View<const T> alpha,View<T> x,T d,T a,T tolerance){
        if(!(d>=0&&d<=1&&a>=0&&a<=1)||(p.opaque&&d!=1))return false;
        const auto& m=model[1];T A=T(m.X)*a*d+T(m.Y)*a*(1-d)+T(m.Z)*d*(1-a),den=A>T(1e-12)?A:T(1),u=a*d/den,v=T(m.Y)*a*(1-d)/den,w=T(m.Z)*d*(1-a)/den;
        if(!(u>=0&&w>=0&&std::isfinite(u)&&std::isfinite(v)&&std::isfinite(w)))return false;
        T minimum=target[0],maximum=target[0];for(int r=1;r<3;++r){if(target[r]<minimum)minimum=target[r];if(target[r]>maximum)maximum=target[r];}
        T preferred=(T(m.mean[0])+T(m.mean[1])+T(m.mean[2]))/3;
        for(int region=0;region<(m.mode==8?2:1);++region){
            T slope[3],bias[3]={0,0,0};int count=2;
            if(m.mode==0){slope[0]=slope[1]=u+v;bias[1]=w;}
            else if(m.mode==1||m.mode==4){slope[0]=v;slope[1]=u+v;bias[1]=w;}
            else if(m.mode==2||m.mode==5){slope[0]=u+v;slope[1]=v;bias[1]=u+w;}
            else if(m.mode==3||m.mode==6||m.mode==7||m.mode==9){slope[0]=slope[1]=v;bias[1]=u+w;}
            else if(m.mode==8){slope[0]=region?2*u+v:v;bias[0]=region?-u:T(0);slope[1]=region?v:2*u+v;bias[1]=region?u+w:w;}
            else if(m.mode==10||m.mode==11){slope[0]=u+v;slope[1]=v-u;bias[1]=u+w;if(m.mode==10){slope[2]=v+w;count=3;}}
            else if(m.mode==12){slope[0]=slope[1]=u+v;bias[1]=u+w;}
            else return false;
            for(int lower=0;lower<count;++lower)for(int upper=0;upper<count;++upper){
                T lo=m.mode==8&&region?T(.5):T(0),hi=m.mode==8&&!region?T(.5):T(1);
                if(!restrict_line(slope[lower],bias[lower],minimum,lo,hi)||!restrict_line(-slope[upper],-bias[upper],-maximum,lo,hi))continue;
                for(int guess=0;guess<3;++guess){T s=guess==0?(preferred<lo?lo:(preferred>hi?hi:preferred)):(guess==1?(lo+hi)/2:(lo+3*hi)/4);bool valid=true;
                    x[0]=d;x[1]=a;for(int r=0;r<3;++r)x[5+r]=a==0&&!m.gray?configured::clip(T(m.mean[r])):s;
                    for(int r=0;r<3&&valid;++r){T bg;valid=invert(m,a,d,A,s,target[r],configured::clip(T(model[0].mean[r])),tolerance,bg);if(valid)x[2+r]=bg;}
                    if(valid&&ConstraintRestore<T,Model>::norm(model,p,target,alpha,{x.data,x.stride})<=tolerance*tolerance*T(.0025))return true;
                }
            }
        }
        return false;
    }
    HD static GRAY_SEED_OUTLINED bool run(const Model* model,Problem p,const T* target,View<const T> alpha,View<T> x,T tolerance){
        if(p.layers!=2||model[0].gray||(p.refinement&&p.smooth))return false;
        for(int r=0;r<3;++r)if(!std::isfinite(target[r]))return false;
        if(p.refinement)return fixed_alpha(model,p,target,alpha,x,alpha[0],alpha[1],tolerance);
        T X=T(model[1].X),Y=T(model[1].Y),Z=T(model[1].Z);
        for(int candidate=0;candidate<(p.opaque?3:8);++candidate){T d=0,a=0,den;
            if(p.opaque){d=1;den=X-Z;if(den!=0){if(candidate)continue;a=(1-Z)/den;}else{if(Z!=1)continue;a=T(candidate)/2;}}
            else if(candidate==0){if(Z==0)continue;d=1/Z;a=0;}
            else if(candidate==1){if(Y==0)continue;d=0;a=1/Y;}
            else if(candidate==2){d=1;den=X-Z;if(den==0){if(Z!=1)continue;a=T(.5);}else a=(1-Z)/den;}
            else if(candidate==3){a=1;den=X-Y;if(den==0){if(Y!=1)continue;d=1;}else d=(1-Y)/den;}
            else if(candidate<7){d=T(candidate-3)/4;den=X*d+Y*(1-d)-Z*d;if(den==0)continue;a=(1-Z*d)/den;}
            else{a=T(.5);den=X*a-Y*a+Z*(1-a);if(den==0)continue;d=(1-Y*a)/den;}
            if(fixed_alpha(model,p,target,alpha,x,d,a,tolerance))return true;
        }
        return false;
    }
};
#undef GRAY_SEED_OUTLINED
}
