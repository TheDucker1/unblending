#pragma once
#include "reachable_range_qp.cuh"

namespace range_union {
using general::View;
// At most one input interval crosses each layer's native discontinuity.
// Continuous images preserve interval count; a Dodge/Burn layer adds at most
// one. Allocate l+1 intervals for prefix l, with explicit capacity checks.
HD inline size_t words(int L){return size_t(3)*L*L+size_t(32)*L;}
template<class T>HD T epsilon(){if constexpr(sizeof(T)==4)return T(firstpass32::blend_function_internal_epsilon);else return T(firstpass::blend_function_internal_epsilon);}
template<class T>HD T cutoff(int mode){
    T eps=epsilon<T>();if(mode==6)return eps;
    // First representable d for the ACTUAL native `1-d < epsilon` predicate.
    T cut=T(1)-eps;
    if(T(1)-cut<eps){T previous=std::nextafter(cut,-T(INFINITY));if(T(1)-previous<eps)cut=previous;}
    else cut=std::nextafter(cut,T(INFINITY));
    return cut;
}
// Exactly equivalent saturation tests avoid a positive rational division
// whose rounded quotient would be clipped to one. Preserve the original
// epsilon checks and delegate unsaturated/NaN cases to the original function.
template<class T>HD T native_blend(T source,T destination,int mode){
    if(mode==6){if(destination<epsilon<T>())return 0;T denominator=T(1)-source;if(denominator<epsilon<T>())return 1;if(destination>=denominator)return 1;}
    else if(mode==7){T numerator=T(1)-destination;if(numerator<epsilon<T>())return 1;if(source<epsilon<T>())return 0;if(numerator>=source)return 0;}
    return general::blend(source,destination,mode);
}
// Native arithmetic order matters at the epsilon cuts: factoring a division
// or moving a product can shift an intermediate prefix across the branch.
template<class T,class Model>HD T native_color(const Model& m,T a,T da,T next,T source,T destination){
    T pre=native_blend(source,destination,m.mode)*a*da+T(m.Y)*a*(T(1)-da)*source+T(m.Z)*da*(T(1)-a)*destination;
    return next>T(1e-12)?pre/next:pre;
}
template<class T>struct Prefix {
    View<T> work;int L,l,r;
    HD int count()const{return int(work[26*L+3*l+r]);}
    HD void count(int n){work[26*L+3*l+r]=T(n);}
    HD int offset(int k)const{return 29*L+3*l*(l+1)+2*r*(l+1)+2*k;}
    HD T low(int k)const{return work[offset(k)];}
    HD T high(int k)const{return work[offset(k)+1];}
    HD void set(int k,T low,T high){work[offset(k)]=low;work[offset(k)+1]=high;}
    HD bool add(T low,T high){
        if(!(std::isfinite(low)&&std::isfinite(high)&&low<=high))return false;
        int n=count(),first=0;while(first<n&&this->high(first)<low)++first;
        int end=first;while(end<n&&this->low(end)<=high){if(this->low(end)<low)low=this->low(end);if(this->high(end)>high)high=this->high(end);++end;}
        if(end==first){if(n==l+1){count(-1);return false;}for(int k=n;k>first;--k)set(k,this->low(k-1),this->high(k-1));set(first,low,high);count(n+1);}
        else{set(first,low,high);int out=first+1;for(int k=end;k<n;++k,++out)set(out,this->low(k),this->high(k));count(out);}
        return true;
    }
    HD bool nearest(T wanted,T& result)const{
        int n=count();if(n<=0||n>l+1)return false;T best=INFINITY;
        for(int k=0;k<n;++k){T low=this->low(k),high=this->high(k),point=wanted<low?low:(wanted>high?high:wanted),distance=std::abs(point-wanted);if(k==0||distance<best){best=distance;result=point;}}
        return true;
    }
};
template<class T>HD bool image(int mode,T low,T high,T u,T v,T w,T& out_low,T& out_high){
    if(mode==6||mode==7){
        if(u<0||v<0||w<0)return false;
        // On either side of its destination epsilon jump, G is continuous
        // and nondecreasing in both variables, even for d outside [0,1].
        out_low=range_refinement::apply(mode,T(0),low,u,v,w);out_high=range_refinement::apply(mode,T(1),high,u,v,w);
        return std::isfinite(out_low)&&std::isfinite(out_high)&&out_low<=out_high;
    }
    if(low<0||high>1){
        // Add/Exclusion may leave the unit cube. Never clamp a prefix: the
        // native blends are still defined there. Source maps are affine on
        // at most two pieces; all extrema occur at their vertices, except
        // SoftLight's source-zero parabola, whose vertex is included below.
        if(mode<0||mode>12)return false;
        if(mode==9&&(u<0||v<0||w<0))return false;
        out_low=INFINITY;out_high=-INFINITY;
        auto probe=[&](T d){
            if(d<low||d>high)return;
            for(int side=0;side<2;++side){T y=range_refinement::apply(mode,T(side),d,u,v,w);if(y<out_low)out_low=y;if(y>out_high)out_high=y;}
            if(mode==8||mode==9){T y=range_refinement::apply(mode,T(.5),d,u,v,w);if(y<out_low)out_low=y;if(y>out_high)out_high=y;}
            if((mode==4||mode==5||mode==10)&&d>=0&&d<=1){T y=range_refinement::apply(mode,d,d,u,v,w);if(y<out_low)out_low=y;if(y>out_high)out_high=y;}
        };
        probe(low);probe(high);
        if(mode==3)probe(T(.5));
        if(mode==4||mode==5||mode==10){probe(T(0));probe(T(1));}
        if(mode==9&&u>0)probe(-w/(2*u));
        return std::isfinite(out_low)&&std::isfinite(out_high)&&out_low<=out_high;
    }
    return range_refinement::reachable(mode,low,high,u,v,w,out_low,out_high);
}
template<class T>HD bool advance(Prefix<T> previous,Prefix<T> next,int mode,T u,T v,T w,T limit=T(INFINITY)){
    next.count(0);T cut=(mode==6||mode==7)?cutoff<T>(mode):T(INFINITY),below=std::nextafter(cut,-T(INFINITY));
    auto add=[&](T l,T h){if(l < -limit)l=-limit;if(h>limit)h=limit;return l>h?true:next.add(l,h);};
    for(int k=0;k<previous.count();++k){T low=previous.low(k),high=previous.high(k),out_low,out_high;
        if((mode==6||mode==7)&&low<cut&&high>=cut){
            if(!image(mode,low,below,u,v,w,out_low,out_high)||!add(out_low,out_high))return false;
            if(!image(mode,cut,high,u,v,w,out_low,out_high)||!add(out_low,out_high))return false;
        }else if(!image(mode,low,high,u,v,w,out_low,out_high)||!add(out_low,out_high))return false;
    }
    return next.count()>0;
}
// A newly reconstructed prefix may differ by a few ulps from its requested
// value. At a discontinuity that tiny error can cause a large output jump.
// Prefer an interior point of the SAME admissible destination interval when
// it is near a Dodge/Burn cut. Never change the reachable union or leave the
// preimage; narrow intervals retain their exact candidate and native checking.
template<class T>struct InteriorNearest {
    T warm,cut,margin,point=0,best=INFINITY;int count=0;
    HD InteriorNearest(T warm,int mode,int depth):warm(warm),cut((mode==6||mode==7)?cutoff<T>(mode):T(INFINITY)),margin(0){
        if(mode==6||mode==7){margin=T(32)*std::numeric_limits<T>::epsilon()*(T(1)+std::abs(cut))*T(depth+1);if(mode==6&&margin>cut/T(8))margin=cut/T(8);}
    }
    HD void add(T low,T high){
        if(!(low<=high))return;T x=warm<low?low:(warm>high?high:warm);
        if(margin>0){if(x<cut&&cut-x<margin&&cut-margin>=low)x=cut-margin;else if(x>=cut&&x-cut<margin&&cut+margin<=high)x=cut+margin;}
        T distance=std::abs(x-warm);if(count==0||distance<best){point=x;best=distance;}++count;
    }
    HD bool nearest(T,T& output)const{if(!count)return false;output=point;return true;}
};
template<class T,class Accumulator>HD void ordered_bracket(Accumulator& out,T low,T high,T target,T p0,T q0,T p1,T q1){
    if(low>high)return;if(range_refinement::inequality(p0,q0,target,true,low,high)&&range_refinement::inequality(p1,q1,target,false,low,high))out.add(low,high);
}
// Invert u*g(d)+w*d+v on the WHOLE real destination domain for nonnegative
// weights. For d<=1/4 put r=1/4-d: g(d)=1/2-r-16r^3. The resulting equation
// 16u*r^3+(u+w)*r=z is strictly monotone. Starting from the smaller of its
// linear/cubic upper bounds gives <=1.466 times the true root; six Newton
// rounds suffice at ordinary precision, with no data-dependent convergence.
template<class T>HD T soft_inverse_wide(T target,T u,T v,T w){
    T y=target-v,cut=T(.5)*u+T(.25)*w;
    if(y<=cut){
        T z=cut-y,linear=u+w;if(!(linear>0))return T(.25);
        T r=z/linear;
        if(u>0){T cubic=std::cbrt(z/(16*u));if(cubic<r)r=cubic;
            #pragma unroll
            for(int i=0;i<6;++i){T cubic_slope=16*u*r*r;r-=(r*(cubic_slope+linear)-z)/(3*cubic_slope+linear);}}
        return T(.25)-r;
    }
    T denominator=u+configured::root(u*u+4*w*y),q=denominator>0?2*y/denominator:T(0);return q*q;
}
template<class T,class Accumulator>HD void soft_destinations_wide(Accumulator& out,T low,T high,T target,T u,T v,T w){
    // A source traverses G0=u*d^2+w*d, Ghalf=(u+w)*d+v/2,
    // G1=u*g(d)+w*d+v. Membership changes only where one equals target:
    // <=2 quadratic roots, one affine root and one monotone cubic root.
    // Six sorted endpoints give at most five intervals; no layer-sized state.
    T cuts[6];int count=2;cuts[0]=low;cuts[1]=high;
    auto insert=[&](T x){if(!std::isfinite(x)||x<=low||x>=high)return;int j=count;while(j>0&&cuts[j-1]>x){cuts[j]=cuts[j-1];--j;}cuts[j]=x;++count;};
    if(u>0){T discriminant=w*w+4*u*target;
        T eps=T(64)*std::numeric_limits<T>::epsilon()*(T(1)+w*w+std::abs(4*u*target));
        if(discriminant>=-eps){if(discriminant<0)discriminant=0;T q=-T(.5)*(w+configured::root(discriminant));if(q!=0){insert(q/u);insert(-target/q);}else insert(T(0));}}
    else if(w>0)insert(target/w);
    if(u+w>0){insert((target-T(.5)*v)/(u+w));insert(soft_inverse_wide(target,u,v,w));}
    auto feasible=[&](T d){T a=range_refinement::apply(9,T(0),d,u,v,w),b=range_refinement::apply(9,T(.5),d,u,v,w),c=range_refinement::apply(9,T(1),d,u,v,w);T l=a<b?a:b,h=a>b?a:b;if(c<l)l=c;if(c>h)h=c;T eps=T(64)*std::numeric_limits<T>::epsilon()*(T(1)+std::abs(target)+std::abs(l)+std::abs(h));return l<=target+eps&&h>=target-eps;};
    for(int k=0;k<count;++k){if(feasible(cuts[k]))out.add(cuts[k],cuts[k]);if(k+1<count&&feasible(cuts[k]+(cuts[k+1]-cuts[k])*T(.5)))out.add(cuts[k],cuts[k+1]);}
}
template<class T,class Accumulator>HD void destinations(Accumulator& out,int mode,T low,T high,T target,T u,T v,T w){
    if(mode==6){
        T cut=cutoff<T>(6),below=std::nextafter(cut,-T(INFINITY));
        if(low<=below)ordered_bracket(out,low,high<below?high:below,target,w,T(0),w,v);
        if(high>=cut&&low<=1)ordered_bracket(out,low>cut?low:cut,high<1?high:T(1),target,u+w,T(0),w,u+v);
        if(high>=1)ordered_bracket(out,low>1?low:T(1),high,target,w,u,w,u+v);
    }else if(mode==7){
        T cut=cutoff<T>(7),below=std::nextafter(cut,-T(INFINITY));
        if(low<=0)ordered_bracket(out,low,high<0?high:T(0),target,w,T(0),w,v);
        if(high>=0&&low<=below)ordered_bracket(out,low>0?low:T(0),high<below?high:below,target,w,T(0),u+w,v);
        if(high>=cut)ordered_bracket(out,low>cut?low:cut,high,target,w,u,w,u+v);
    }else if((low<0||high>1)&&(mode==4||mode==5)){
        if(low<=0){T h=high<0?high:T(0);if(mode==4)ordered_bracket(out,low,h,target,u+w,T(0),u+w,v);else ordered_bracket(out,low,h,target,w,T(0),w,u+v);}
        T l=low>0?low:T(0),h=high<1?high:T(1);if(l<=h){if(mode==4)ordered_bracket(out,l,h,target,w,T(0),u+w,v);else ordered_bracket(out,l,h,target,u+w,T(0),w,u+v);}
        if(high>=1){l=low>1?low:T(1);if(mode==4)ordered_bracket(out,l,high,target,w,T(0),w,u+v);else ordered_bracket(out,l,high,target,u+w,T(0),u+w,v);}
    }else if((low<0||high>1)&&mode==8){
        range_refinement::bracket(out,low,high,target,w,T(0),u+w,T(.5)*v);
        range_refinement::bracket(out,low,high,target,u+w,T(.5)*v,w,u+v);
    }else if((low<0||high>1)&&mode==10){
        if(low<=0)ordered_bracket(out,low,high<0?high:T(0),target,w-u,T(0),w-u,u+v);
        T l=low>0?low:T(0),h=high<1?high:T(1);if(l<=h)out=range_refinement::destination_set(mode,l,h,target,u,v,w,out);
        if(high>=1)range_refinement::bracket(out,low>1?low:T(1),high,target,u+w,T(0),u+w,v-u);
    }else if((low<0||high>1)&&mode==9)soft_destinations_wide(out,low,high,target,u,v,w);
    else out=range_refinement::destination_set(mode,low,high,target,u,v,w,out);
}
template<class T,class Accumulator>HD bool exact_warm(Accumulator&,int,T,T,T,T,T){return false;}
template<class T,int Capacity>HD bool exact_warm(range_refinement::Nearest<T,Capacity>& out,int mode,T d,T target,T u,T v,T w){
    T value=u==0&&v==0?w*d:u*native_blend(out.warm,d,mode)+v*out.warm+w*d;
    if(value!=target)return false;out.add(out.warm,out.warm);return true;
}
// Optional GPU solver arithmetic only. Original-double rendering/validation
// never uses this approximate reciprocal path. CPU controls retain hardware
// division, so they are not slowed down to imitate GPU arithmetic.
template<class T>HD T solver_divide(T numerator,T denominator){
#if defined(UNBLENDING_APPROX_SOLVER) && defined(__CUDA_ARCH__)
    if constexpr(std::is_same_v<T,float>)return __fdividef(numerator,denominator);
#endif
    return numerator/denominator;
}
// Rational piece inversion with the linear source term v*s retained.
// Saturated plateaus stay intervals; the Gaussian candidate may use their
// freedom. Stable quadratic roots avoid cancellation at small u or v.
template<class T,class Accumulator>HD Accumulator sources(Accumulator out,int mode,T d,T target,T u,T v,T w){
    if(mode!=6&&mode!=7)return range_refinement::source_set(mode,d,target,u,v,w,out);
    if(exact_warm(out,mode,d,target,u,v,w))return out;
    bool flat=u==0||(mode==6?(d<epsilon<T>()||d>=1):(d<=0||T(1)-d<epsilon<T>()));
    if(flat){range_refinement::inverse_segment(out,T(0),T(1),range_refinement::apply(mode,T(0),d,u,v,w),range_refinement::apply(mode,T(1),d,u,v,w),target);return out;}
    T split=T(1)-d,left=mode==6?T(0):split,right=mode==6?split:T(1);
    if(mode==6)range_refinement::inverse_segment(out,split,T(1),range_refinement::apply(mode,split,d,u,v,w),range_refinement::apply(mode,T(1),d,u,v,w),target);
    else range_refinement::inverse_segment(out,T(0),split,range_refinement::apply(mode,T(0),d,u,v,w),range_refinement::apply(mode,split,d,u,v,w),target);
    T lo=range_refinement::apply(mode,left,d,u,v,w),hi=range_refinement::apply(mode,right,d,u,v,w),eps=T(64)*std::numeric_limits<T>::epsilon()*(T(1)+std::abs(target)+std::abs(lo)+std::abs(hi));
    if(target<lo-eps||target>hi+eps)return out;
    T value;
    if(mode==6){T y=target-w*d,c=y-u*d;if(v==0)value=y!=0?T(1)-solver_divide(u*d,y):T(0);else{T b=v+y,root=configured::root((v-y)*(v-y)+4*v*u*d),den=b+root;value=den!=0?solver_divide(2*c,den):T(0);}}
    else {T y=target-u-w*d,c=u*(T(1)-d);if(v==0)value=y!=0?solver_divide(-c,y):T(1);else{T root=configured::root(y*y+4*v*c);value=y>0?solver_divide(y+root,2*v):(root-y!=0?solver_divide(2*c,root-y):T(0));}}
    value=value<left?left:(value>right?right:value);if(std::isfinite(value))out.add(value,value);return out;
}

template<class T,class Input=T,class Output=T,class Model=general::Layer>HD int pixel(const Model* models,general::Problem p,
    View<const Input> initial,View<const Input> target,View<const Input> alpha,View<const Input> background,
    View<Output> output,View<T> work,T tolerance,T anchor=T(0),bool reuse=false,T limit=T(INFINITY)){
    int L=p.layers;bool so=range_refinement::opaque_source_over(models,p)&&alpha[0]==1;
    auto warm=[&](int l,int r){T x=T(initial[L+3*l+r]);if(anchor!=0)x+=anchor*(T(models[l].mean[r])-x);return configured::clip(x);};
    for(int l=0;l<L;++l){Input a=alpha[l];if(!std::isfinite(a)||a<0||a>1||(p.opaque&&l==0&&a!=1))return 4;output[l]=a;for(int r=0;r<3;++r)output[L+3*l+r]=warm(l,r);if(models[l].gray)return 8;}
    T current[3];for(int r=0;r<3;++r){Input bg=background[r];if(!std::isfinite(bg)||bg<0||bg>1)return 4;current[r]=p.smooth?T(bg):warm(0,r);if(p.smooth)output[L+r]=bg;
        if(!reuse){Prefix<T> prefix{work,L,0,r};prefix.count(0);if(!prefix.add(p.smooth?T(bg):T(0),p.smooth?T(bg):T(1)))return 9;work[r]=prefix.low(0);work[3*L+r]=prefix.high(0);}work[6*L+r]=current[r];}
    T da=T(alpha[0]);if(!reuse)work[9*L]=da;
    for(int l=1;l<L;++l){T next,u,v,w;if(!range_refinement::weights(models[l],T(alpha[l]),da,next,u,v,w,so))return 8;if((models[l].mode==6||models[l].mode==7)&&(u<0||v<0||w<0))return 8;
        if(!reuse)work[9*L+l]=next;
        for(int r=0;r<3;++r){if(!reuse){Prefix<T> prefix{work,L,l,r},previous{work,L,l-1,r};if(!advance(previous,prefix,models[l].mode,u,v,w,limit))return prefix.count()<0?9:(std::isfinite(limit)&&prefix.count()==0?6:8);work[3*l+r]=prefix.low(0);work[3*L+3*l+r]=prefix.high(prefix.count()-1);}
            current[r]=native_color(models[l],T(alpha[l]),da,next,warm(l,r),current[r]);work[6*L+3*l+r]=current[r];}da=next;}
    T goal[3],desired[3];bool projected=false;
    for(int r=0;r<3;++r){T wanted=T(target[r]);if(!std::isfinite(wanted))return 4;Prefix<T> last{work,L,L-1,r};if(!last.nearest(wanted,goal[r]))return 9;desired[r]=goal[r];projected|=goal[r]!=wanted;}
    unsigned active=7;
    for(int l=L-1;l>0&&active;--l){T next,u,v,w;if(!range_refinement::weights(models[l],T(alpha[l]),so?T(1):work[9*L+l-1],next,u,v,w,so))return 8;
        for(int r=0;r<3;++r){if(!(active&(1u<<r)))continue;T destination=work[6*L+3*(l-1)+r],color;
            // The feasible source/destination pair is underdetermined. Large
            // prefixes can make an affine source inverse catastrophically
            // sensitive (SoftLight grows cubically for negative d). Prefer
            // the nearest feasible destination to the warm prefix clipped to
            // [0,1]. This projects a PREFERENCE, never the prefix itself or
            // its reachable set; if no unit destination is feasible, retain
            // the closest outside value. Do not short-circuit through the
            // ill-conditioned warm source inverse in that case.
            T preferred=std::isfinite(destination)?configured::clip(destination):T(.5);bool conditioned=preferred!=destination;
            if(!conditioned){auto warm_sources=sources(range_refinement::Nearest<T,2>{warm(l,r)},models[l].mode,destination,desired[r],u,v,w);
                if(warm_sources.nearest(warm(l,r),color)){output[L+3*l+r]=configured::clip(color);active&=~(1u<<r);continue;}}
            InteriorNearest<T> nearest(preferred,models[l].mode,l);Prefix<T> previous{work,L,l-1,r};
            for(int k=0;k<previous.count();++k)destinations(nearest,models[l].mode,previous.low(k),previous.high(k),desired[r],u,v,w);
            if(!nearest.nearest(destination,destination))return 2;
            auto inverse=sources(range_refinement::Nearest<T,2>{warm(l,r)},models[l].mode,destination,desired[r],u,v,w);if(!inverse.nearest(warm(l,r),color))return 2;
            output[L+3*l+r]=configured::clip(color);desired[r]=destination;}
    }
    if(!p.smooth)for(int r=0;r<3;++r)if(active&(1u<<r))output[L+r]=configured::clip(desired[r]);
    for(int r=0;r<3;++r)current[r]=T(output[L+r]);da=T(alpha[0]);
    for(int l=1;l<L;++l){T next,u,v,w;if(!range_refinement::weights(models[l],T(alpha[l]),da,next,u,v,w,so))return 8;for(int r=0;r<3;++r)current[r]=native_color(models[l],T(alpha[l]),da,next,T(output[L+3*l+r]),current[r]);da=next;}
    T error=0;for(int r=0;r<3;++r){T delta=current[r]-goal[r];error+=delta*delta;}if(!std::isfinite(error)||error>tolerance*tolerance)return 3;return projected?5:0;
}
template<class T,class Model>HD int improve(const Model* model,general::Problem p,View<const T> initial,View<const T> target,View<const T> alpha,View<const T> bg,View<T> point,View<T> work,T tolerance,T anchor,int rounds,int& accepted,T limit=T(INFINITY)){
    int L=p.layers;accepted=0;int status=pixel(model,p,initial,target,alpha,bg,point,work,tolerance,anchor,false,limit);if(status!=0&&status!=5)return status;if(rounds==0)return status;
    T goal[3];for(int r=0;r<3;++r){Prefix<T> last{work,L,L-1,r};if(!last.nearest(target[r],goal[r]))return 9;}
    T best=range_refinement::gaussian_energy(model,p,point);if(!std::isfinite(best))return 2;View<T> trial{work.data+size_t(22*L)*work.stride,work.stride};
    for(int it=0;it<rounds;++it){if(!range_refinement::proposal_views(model,p,View<const T>{goal,1},point,work))break;
        int next=pixel(model,p,View<const T>{work.data+size_t(18*L)*work.stride,work.stride},target,alpha,bg,trial,work,tolerance,T(0),true,limit);T cost=range_refinement::gaussian_energy(model,p,trial);
        if((next==0||next==5)&&std::isfinite(cost)&&cost<best){for(int k=0;k<4*L;++k)point[k]=trial[k];best=cost;status=next;++accepted;}else break;
    }
    return status;
}
template<class T,class Model,bool Refine>__global__ void kernel(const Model* model,general::Problem p,int n,int slots,const T* initial,const T* target,const T* alpha,const T* bg,T* output,T* state,int* statuses,int* accepted,T tolerance,T anchor,int rounds,T* ranges,int* counts,T* intervals,bool dump){
    int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=slots)return;View<T> work{state+slot,size_t(slots)};
    for(int i=slot;i<n;i+=slots){auto cv=[&](const T* x){return View<const T>{x+i,size_t(n)};};View<T> point{output+i,size_t(n)};
        int status;if constexpr(Refine)status=improve(model,p,cv(initial),cv(target),cv(alpha),cv(bg),point,work,tolerance,anchor,rounds,accepted[i]);else{status=pixel(model,p,cv(initial),cv(target),cv(alpha),cv(bg),point,work,tolerance,anchor);accepted[i]=0;}statuses[i]=status;
        for(int r=0;r<3;++r){Prefix<T> last{work,p.layers,p.layers-1,r};int count=(status==0||status==5||status==3)?last.count():0;counts[size_t(r)*n+i]=count;ranges[size_t(r)*n+i]=count?last.low(0):T(0);ranges[size_t(3+r)*n+i]=count?last.high(count-1):T(0);
            if(dump)for(int k=0;k<p.layers;++k){intervals[size_t(2*(r*p.layers+k))*n+i]=k<count?last.low(k):T(0);intervals[size_t(2*(r*p.layers+k)+1)*n+i]=k<count?last.high(k):T(0);}}
    }
}
}
