#pragma once
#include "model/pixel_problem.cuh"
#include "solver/two_layer_inverse.cuh"
#include <limits>

namespace range_refinement {
// Candidate: preserve a warm prefix when possible, move it only far enough
// to admit the final target. This is a feasible reconstruction algorithm,
// not a claim of globally minimum Gaussian energy.
// Distinct proven capacities: at most two source pieces, four destination
// pieces (Overlay's two destination branches, each at most two intervals).
// Overflow invalidates the set instead of silently discarding a solution.
template<class T,int Capacity>struct Set {
    T lo[Capacity],hi[Capacity];int count=0;
    HD void add(T l,T h){
        if(!(l<=h)||count<0)return;if(count==Capacity){count=-1;return;}
        #pragma unroll
        for(int i=0;i<Capacity;++i)if(count==i){lo[i]=l;hi[i]=h;}
        ++count;
    }
    HD bool nearest(T warm,T& result)const{
        if(count<0)return false;T best=INFINITY;bool found=false;
        #pragma unroll
        for(int i=0;i<Capacity;++i)if(i<count){T x=warm<lo[i]?lo[i]:(warm>hi[i]?hi[i]:warm),distance=std::abs(x-warm);if(!found||distance<best){found=true;best=distance;result=x;}}
        return found;
    }
};
// The only consumer needs the nearest point in the union. Project each
// interval immediately; min over interval projections equals projection onto
// their union. Preserve first-interval ties and detect capacity violations.
template<class T,int Capacity>struct Nearest {
    T warm,point=0,best=INFINITY;int count=0;
    HD void add(T l,T h){
        if(!(l<=h)||count<0)return;if(count==Capacity){count=-1;return;}
        T x=warm<l?l:(warm>h?h:warm),distance=std::abs(x-warm);
        if(count==0||distance<best){point=x;best=distance;}++count;
    }
    HD bool nearest(T,T& result)const{if(count<=0)return false;result=point;return true;}
};
template<class Model>HD bool opaque_source_over(const Model* models,const general::Problem& p){
    if constexpr(std::is_same_v<Model,gaussian_box::Model>)return p.opaque&&models[0].opaque_source_over;
    else return false;
}
template<class T>HD T apply(int mode,T source,T destination,T u,T v,T w){
    // Zero source coefficients make the source/color operation unobservable.
    // Keep w: a transparent zero-alpha prefix need not pass through unchanged.
    if(u==0&&v==0)return w*destination;
    return u*general::blend(source,destination,mode)+v*source+w*destination;
}
template<class T>HD bool corner_extrema_mode(int mode){return mode==0||mode==1||mode==2||mode==11||mode==12;}
template<class T>HD bool unit_mode(int mode){return mode>=0&&mode<=12&&mode!=6&&mode!=7;}
template<class T>HD bool weights(const general::Layer& m,T a,T destination_alpha,T& next,T& u,T& v,T& w,bool opaque_so=false){
    if(opaque_so){next=1;u=a;v=0;w=1-a;return true;}
    next=T(m.X)*a*destination_alpha+T(m.Y)*a*(1-destination_alpha)+T(m.Z)*destination_alpha*(1-a);
    T den=next>T(1e-12)?next:T(1);u=a*destination_alpha/den;v=T(m.Y)*a*(1-destination_alpha)/den;w=T(m.Z)*destination_alpha*(1-a)/den;
    T eps=T(64)*std::numeric_limits<T>::epsilon()*(T(1)+std::abs(u)+std::abs(v)+std::abs(w));
    return configured::isfinite(next)&&configured::isfinite(u)&&configured::isfinite(v)&&configured::isfinite(w)&&u>=-eps&&v>=-eps&&w>=-eps;
}
template<class T>HD bool reachable(int mode,T low,T high,T u,T v,T w,T& out_low,T& out_high){
    T eps=T(64)*std::numeric_limits<T>::epsilon()*(T(1)+std::abs(low)+std::abs(high));
    if(!unit_mode<T>(mode)||(!corner_extrema_mode<T>(mode)&&(low<-eps||high>T(1)+eps)))return false;
    out_low=INFINITY;out_high=-INFINITY;
    for(int endpoint=0;endpoint<2;++endpoint){T d=endpoint?high:low;
        for(int side=0;side<2;++side){T value=apply(mode,T(side),d,u,v,w);if(value<out_low)out_low=value;if(value>out_high)out_high=value;}
        if(mode==10&&d>=0&&d<=1){T value=(v+w)*d;if(value<out_low)out_low=value;if(value>out_high)out_high=value;}
    }
    return configured::isfinite(out_low)&&configured::isfinite(out_high);
}
// Intersect p*d+q <= target (or >= target) with an existing interval.
template<class T>HD bool inequality(T p,T q,T target,bool below,T& low,T& high){
    if(!below){p=-p;q=-q;target=-target;}
    T magnitude=std::abs(low)>std::abs(high)?std::abs(low):std::abs(high);
    T eps=T(32)*std::numeric_limits<T>::epsilon()*(T(1)+std::abs(target)+std::abs(q)+std::abs(p)*magnitude);
    if(p>0){T x=(target-q)/p;if(x<low){if(p*low+q>target+eps)return false;x=low;}if(x<high)high=x;}
    else if(p<0){T x=(target-q)/p;if(x>high){if(p*high+q>target+eps)return false;x=high;}if(x>low)low=x;}
    else if(q>target+eps)return false;
    return low<=high;
}
template<class T,class Accumulator>HD void bracket(Accumulator& out,T low,T high,T target,T p0,T q0,T p1,T q1){
    T l=low,h=high;if(inequality(p0,q0,target,true,l,h)&&inequality(p1,q1,target,false,l,h))out.add(l,h);
    l=low;h=high;if(inequality(p1,q1,target,true,l,h)&&inequality(p0,q0,target,false,l,h))out.add(l,h);
}
template<class T>HD T soft_bound(int source,T d,T u,T v,T w){return apply(9,T(source),d,u,v,w);}
// Stable inverses of SoftLight's monotone source-endpoint maps. The lower
// map is quadratic. Above d=.25 the upper map is quadratic in sqrt(d).
// Below .25 its normalized cubic is concave and increasing; Newton from the
// linear lower estimate converges monotonically (six bounded rounds).
template<class T>HD T soft_inverse0(T target,T u,T w){
    if(target<=0)return 0;if(u==0)return w>0?target/w:T(0);
    T den=w+configured::root(w*w+4*u*target);return den>0?2*target/den:T(0);
}
template<class T>HD T soft_inverse1(T target,T u,T v,T w){
    T z=target-v;if(z<=0)return 0;if(z>=u+w)return 1;
    if(z>=T(.5)*u+T(.25)*w){T den=u+configured::root(u*u+4*w*z),q=den>0?2*z/den:T(0);return q*q;}
    T scale=4*u+w;if(!(scale>0))return 0;T ratio=u/scale,rhs=z/scale,d=rhs;
    #pragma unroll
    for(int i=0;i<6;++i){T value=d+ratio*(16*d-12)*d*d-rhs,slope=1+ratio*(48*d-24)*d;d-=value/slope;}
    return d;
}
template<class T,class Accumulator=Set<T,4>>HD Accumulator destination_set(int mode,T low,T high,T target,T u,T v,T w,Accumulator out={}){
    switch(mode){
    case 0:case 8:bracket(out,low,high,target,w,T(0),w,u+v);break;
    case 1:case 4:bracket(out,low,high,target,w,T(0),u+w,v);break;
    case 2:case 5:bracket(out,low,high,target,u+w,T(0),w,u+v);break;
    case 12:bracket(out,low,high,target,u+w,T(0),u+w,u+v);break;
    case 11:bracket(out,low,high,target,u+w,T(0),w-u,u+v);break;
    case 3:{T middle=T(.5);if(low<=middle)bracket(out,low,high<middle?high:middle,target,w,T(0),2*u+w,v);if(high>=middle)bracket(out,low>middle?low:middle,high,target,2*u+w,-u,w,u+v);break;}
    case 10:{
        // G(s,d)=u*abs(s-d)+v*s+w*d. For |v|>=|u| it is
        // monotone in s. Otherwise its sole interior source extremum is s=d.
        if(v>=std::abs(u)||v<=-std::abs(u)){bracket(out,low,high,target,u+w,T(0),w-u,u+v);break;}
        const T endpoint_p[2]={u+w,w-u},endpoint_q[2]={0,u+v};
        for(int j=0;j<2;++j){T l=low,h=high;
            bool ok=u>=0?(inequality(v+w,T(0),target,true,l,h)&&inequality(endpoint_p[j],endpoint_q[j],target,false,l,h)):
                (inequality(endpoint_p[j],endpoint_q[j],target,true,l,h)&&inequality(v+w,T(0),target,false,l,h));
            if(ok)out.add(l,h);
        }break;}
    case 9:{
        // Both endpoint functions are monotone on a unit destination range.
        // Use analytic endpoint inverses for nonnegative weights; retain native
        // bisection for signed coefficients admitted only as rounding noise.
        if(u==0){bracket(out,low,high,target,w,T(0),w,v);break;}
        T lower=low,upper=high;
        T minimum=soft_bound(0,low,u,v,w),maximum=soft_bound(1,high,u,v,w);
        T eps=T(64)*std::numeric_limits<T>::epsilon()*(T(1)+std::abs(target)+std::abs(minimum)+std::abs(maximum));
        if(maximum<target-eps||minimum>target+eps)break;
        target=target<minimum?minimum:(target>maximum?maximum:target);
        bool closed=u>=0&&w>=0;
        if(closed){
            if(soft_bound(1,low,u,v,w)<target)lower=soft_inverse1(target,u,v,w);
            if(soft_bound(0,high,u,v,w)>target)upper=soft_inverse0(target,u,w);
            lower=lower<low?low:(lower>high?high:lower);upper=upper<low?low:(upper>high?high:upper);
        }else{
        if(soft_bound(1,low,u,v,w)<target){T l=low,h=high;
            #pragma unroll 1
            for(int it=0;it<(sizeof(T)==4?26:54);++it){T mid=(l+h)*T(.5);if(soft_bound(1,mid,u,v,w)<target)l=mid;else h=mid;}lower=h;}
        if(soft_bound(0,high,u,v,w)>target){T l=low,h=high;
            #pragma unroll 1
            for(int it=0;it<(sizeof(T)==4?26:54);++it){T mid=(l+h)*T(.5);if(soft_bound(0,mid,u,v,w)>target)h=mid;else l=mid;}upper=l;}
        }
        T resolution=closed?T(0):(high-low)/(sizeof(T)==4?T(67108864):T(18014398509481984.));
        if(lower>upper&&lower-upper<=resolution+eps){T point=(lower+upper)*T(.5);lower=upper=point;}
        out.add(lower,upper);break;}
    }
    return out;
}
// Source G is affine except piecewise blends. General v*s changes inverse
// slopes, so the old opaque two-layer blend inverse alone is insufficient.
template<class T,class Accumulator>HD void inverse_segment(Accumulator& out,T left,T right,T at_left,T at_right,T target){
    T lo=at_left<at_right?at_left:at_right,hi=at_left>at_right?at_left:at_right;
    T eps=T(64)*std::numeric_limits<T>::epsilon()*(T(1)+std::abs(target)+std::abs(lo)+std::abs(hi));
    if(target<lo-eps||target>hi+eps)return;
    if(at_left==at_right){out.add(left,right);return;}
    T value=left+(right-left)*(target-at_left)/(at_right-at_left);value=value<left?left:(value>right?right:value);out.add(value,value);
}
template<class T,class Accumulator=Set<T,2>>HD Accumulator source_set(int mode,T d,T target,T u,T v,T w,Accumulator out={}){
    T cut=1;int pieces=1;
    if(mode==4||mode==5||mode==10){if(d>0&&d<1){cut=d;pieces=2;}}
    else if(mode==8||mode==9){cut=T(.5);pieces=2;}
    for(int part=0;part<pieces;++part){T left=part?cut:T(0),right=part+1==pieces?T(1):cut;inverse_segment(out,left,right,apply(mode,left,d,u,v,w),apply(mode,right,d,u,v,w),target);}
    return out;
}
// Scratch: [low3L, high3L, warm3L, prefixAlphaL]. State and image strides
// are independent, allowing bounded reusable GPU state without pixel repacking.
// All states are linear in layer count; no per-thread layer-sized arrays.
template<class T,class Input=T,class Output=T,class Model=general::Layer>HD int pixel_views(const Model* models,general::Problem p,
    general::View<const Input> initial,general::View<const Input> target,general::View<const Input> alpha,general::View<const Input> background,
    general::View<Output> output,general::View<T> scratch,T tolerance,T anchor=T(0),bool reuse_ranges=false){
    int L=p.layers;bool opaque_so=opaque_source_over(models,p)&&alpha[0]==1;auto warm=[&](int l,int r){T value=T(initial[L+3*l+r]);if(anchor!=0)value+=anchor*(T(models[l].mean[r])-value);return configured::clip(value);};auto at=[&](int plane)->T&{return scratch[plane];};
    for(int l=0;l<L;++l){Input a=alpha[l];if(!configured::isfinite(a)||a<0||a>1||(p.opaque&&l==0&&a!=1))return 4;output[l]=a;
        for(int r=0;r<3;++r)output[L+3*l+r]=warm(l,r);if(models[l].gray)return 8;}
    T current[3];for(int r=0;r<3;++r){Input bg=background[r];if(!configured::isfinite(bg)||bg<0||bg>1)return 4;
        current[r]=p.smooth?T(bg):warm(0,r);if(!reuse_ranges){at(r)=p.smooth?T(bg):T(0);at(3*L+r)=p.smooth?T(bg):T(1);}at(6*L+r)=current[r];if(p.smooth)output[L+r]=bg;}
    T da=T(alpha[0]);at(9*L)=da;
    for(int l=1;l<L;++l){T next,u,v,w;if(!weights(models[l],T(alpha[l]),da,next,u,v,w,opaque_so))return 8;if(!reuse_ranges)at(9*L+l)=next;
        for(int r=0;r<3;++r){if(!reuse_ranges&&!reachable(models[l].mode,at(3*(l-1)+r),at(3*L+3*(l-1)+r),u,v,w,at(3*l+r),at(3*L+3*l+r)))return 8;
            current[r]=apply(models[l].mode,warm(l,r),current[r],u,v,w);at(6*L+3*l+r)=current[r];}da=next;}
    T desired[3],goal[3];bool projected=false;
    for(int r=0;r<3;++r){T value=T(target[r]),low=at(3*(L-1)+r),high=at(3*L+3*(L-1)+r);if(!configured::isfinite(value))return 4;
        desired[r]=goal[r]=value<low?low:(value>high?high:value);projected|=goal[r]!=value;}
    unsigned active_channels=7;
    for(int l=L-1;l>0&&active_channels;--l){T next,u,v,w;if(!weights(models[l],T(alpha[l]),at(9*L+l-1),next,u,v,w,opaque_so))return 8;
        for(int r=0;r<3;++r){if(!(active_channels&(1u<<r)))continue;
            T low=at(3*(l-1)+r),high=at(3*L+3*(l-1)+r),destination=at(6*L+3*(l-1)+r),color;
            // If the warm prefix admits the requested output, preserve it.
            // Every lower warm color ALREADY reconstructs that prefix, so no
            // further backward inversion is needed for this channel.
            auto warm_sources=source_set(models[l].mode,destination,desired[r],u,v,w,Nearest<T,2>{warm(l,r)});
            if(warm_sources.nearest(warm(l,r),color)){output[L+3*l+r]=configured::clip(color);active_channels&=~(1u<<r);continue;}
            auto set=destination_set(models[l].mode,low,high,desired[r],u,v,w,Nearest<T,4>{at(6*L+3*(l-1)+r)});
            if(!set.nearest(at(6*L+3*(l-1)+r),destination))return 2;
            auto sources=source_set(models[l].mode,destination,desired[r],u,v,w,Nearest<T,2>{warm(l,r)});
            if(!sources.nearest(warm(l,r),color))return 2;
            output[L+3*l+r]=configured::clip(color);desired[r]=destination;}
    }
    if(!p.smooth)for(int r=0;r<3;++r)if(active_channels&(1u<<r))output[L+r]=configured::clip(desired[r]);
    // Recompute the ACTUAL native chain, rather than trusting inverse algebra.
    for(int r=0;r<3;++r)current[r]=T(output[L+r]);da=T(alpha[0]);
    for(int l=1;l<L;++l){T next,u,v,w;if(!weights(models[l],T(alpha[l]),da,next,u,v,w,opaque_so))return 8;for(int r=0;r<3;++r)current[r]=apply(models[l].mode,T(output[L+3*l+r]),current[r],u,v,w);da=next;}
    T error=0;for(int r=0;r<3;++r){T delta=current[r]-goal[r];error+=delta*delta;}
    if(!configured::isfinite(error)||error>tolerance*tolerance)return 3;return projected?5:0;
}
// Legacy full-image planes, retained for CPU and untiled GPU controls.
template<class T,class Input=T,class Output=T,class Model=general::Layer>HD int pixel(const Model* models,general::Problem p,int n,int index,
    const Input* initial,const Input* target,const Input* alpha,const Input* background,Output* output,T* scratch,T tolerance,T anchor=T(0),bool reuse_ranges=false){
    return pixel_views(models,p,general::View<const Input>{initial+index,size_t(n)},general::View<const Input>{target+index,size_t(n)},
        general::View<const Input>{alpha+index,size_t(n)},general::View<const Input>{background+index,size_t(n)},
        general::View<Output>{output+index,size_t(n)},general::View<T>{scratch+index,size_t(n)},tolerance,anchor,reuse_ranges);
}
template<class T>__global__ void kernel(const general::Layer* model,general::Problem p,int n,const T* initial,const T* target,const T* alpha,const T* background,T* output,T* scratch,int* status,T tolerance,T anchor){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)status[i]=pixel(model,p,n,i,initial,target,alpha,background,output,scratch,tolerance,anchor);}
}
