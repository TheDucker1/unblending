#pragma once
// Feasible retraction: keep all other trial variables fixed and eliminate one
// nongray source color per output channel. This is a layer-chain inverse, not
// an unconstrained color clamp. The caller validates the original residual.
namespace algebraic_retract {
template<class T>struct Inverse {
    using Nearest=range_refinement::Nearest<T,4>;
    HD static void affine(Nearest& out,T slope,T offset,T lo,T hi,T target){
        if(lo>hi)return;T eps=T(16)*std::numeric_limits<T>::epsilon()*(1+std::abs(target)+std::abs(offset));
        if(slope==0){if(std::abs(target-offset)<=eps)out.add(lo,hi);return;}
        T value=(target-offset)/slope;
        if(value>=lo-eps&&value<=hi+eps)out.add(value<lo?lo:(value>hi?hi:value),value<lo?lo:(value>hi?hi:value));
    }
    HD static T apply(int mode,T a,T s,T d){return a*range_union::native_blend(s,d,mode)+(1-a)*d;}
    HD static void segment(Nearest& out,int mode,T a,T s,T lo,T hi,T target){
        if(lo>hi)return;
        range_refinement::inverse_segment(out,lo,hi,apply(mode,a,s,lo),apply(mode,a,s,hi),target);
    }
    HD static bool destination(int mode,T a,T s,T target,T warm,T& result){
        Nearest out{warm};
        if(mode==6){
            T eps=range_union::epsilon<T>(),before=std::nextafter(eps,-T(INFINITY));
            T w=1-a,split=1-s;affine(out,w,0,0,before,target);
            if(split<eps)affine(out,w,a,eps,1,target);
            else {T end=split<1?split:T(1);affine(out,w+a/split,0,eps,end,target);affine(out,w,a,end,1,target);}
        }else if(mode==7){
            T cut=range_union::cutoff<T>(7),before=std::nextafter(cut,-T(INFINITY)),split=1-s;
            T w=1-a;
            if(s<range_union::epsilon<T>())affine(out,w,0,0,before,target);
            else {
                T end=split<before?split:before;affine(out,w,0,0,end,target);
                // Scale the equation by s to avoid cancellation in a*(1-1/s).
                Nearest scaled{warm};affine(scaled,w*s+a,-a*split,end,before,target*s);T d;
                if(scaled.nearest(warm,d))out.add(d,d);
            }
            affine(out,w,a,cut,1,target);
        }else if(mode==8){
            T slope=s<=T(.5)?2*s:2*(1-s),offset=s<=T(.5)?T(0):2*s-1;
            affine(out,1-a+a*slope,a*offset,0,1,target);
        }else if(mode==0){affine(out,1-a,a*s,0,1,target);
        }else if(mode==1){affine(out,1-a+a*s,0,0,1,target);
        }else if(mode==2){affine(out,1-a+a*(1-s),a*s,0,1,target);
        }else if(mode==3){segment(out,mode,a,s,0,T(.5),target);segment(out,mode,a,s,T(.5),1,target);}
        else if(mode==4||mode==5||mode==10){segment(out,mode,a,s,0,s,target);segment(out,mode,a,s,s,1,target);}
        else if(mode==9){
            // SoftLight is monotone in destination. Its dark-source branch and
            // its light-source branch above .25 are quadratics (in sqrt(d) for
            // the latter); six safeguarded Newton rounds handle the cubic.
            if(s<=T(.5)){
                T p=a*(1-2*s),q=1-a+2*a*s,disc=configured::root(q*q+4*p*target),den=q+disc;
                T d=den>0?2*target/den:0;if(d>=0&&d<=1)out.add(d,d);
            }else if(target>=apply(mode,a,s,T(.25))){
                T p=1-a+a*(2-2*s),q=a*(2*s-1),disc=configured::root(q*q+4*p*target),den=q+disc;
                T z=den>0?2*target/den:0,d=z*z;if(d>=T(.25)&&d<=1)out.add(d,d);
            }else{
                T lo=0,hi=T(.25),d=warm<lo?lo:(warm>hi?hi:warm);
                for(int k=0;k<8;++k){T value=apply(mode,a,s,d)-target;if(value>0)hi=d;else lo=d;
                    T slope=a*general::grad_d(s,d,mode)+(1-a),next=slope>0?d-value/slope:(lo+hi)*T(.5);
                    d=next>=lo&&next<=hi?next:(lo+hi)*T(.5);}
                out.add(d,d);
            }
        }else segment(out,mode,a,s,0,1,target);
        return out.nearest(warm,result)&&configured::isfinite(result);
    }
    HD static bool source(int mode,T a,T d,T target,T warm,T& value){
        if(!(a>0))return false;
        T v=(target-(1-a)*d)/a,eps=T(32)*std::numeric_limits<T>::epsilon();
        if(mode==0)value=v;
        else if(mode==8){
            if(v<-eps||v>1+eps)return false;v=configured::clip(v);
            if(d==0&&v==0)value=warm<T(.5)?warm:T(.5);
            else if(d==1&&v==1)value=warm>T(.5)?warm:T(.5);
            else value=v<=d?v/(2*d):1-(1-v)/(2*(1-d));
        }else if(mode==6){
            if(d<range_union::epsilon<T>()){if(std::abs(v)>eps)return false;value=warm;}
            else {if(v<d-eps||v>1+eps)return false;
                if(v>=1){T low=1-d;value=warm<low?low:warm;}
                else if(v>0)value=1-d/v;else return false;}
        }else if(mode==7){
            if(1-d<range_union::epsilon<T>()){if(std::abs(v-1)>eps)return false;value=warm;}
            else {if(v<-eps||v>d+eps)return false;
                if(v<=0){T high=1-d;value=warm>high?high:warm;}
                else value=(1-d)/(1-v);}
        }else {
            auto inverse=range_union::sources(range_refinement::Nearest<T,2>{warm},mode,d,target,a,T(0),T(1)-a);
            return inverse.nearest(warm,value);
        }
        if(value<-eps||value>1+eps||!configured::isfinite(value))return false;
        value=configured::clip(value);return true;
    }
    template<class Model>HD static bool eligible(const Model* m,general::Problem p,general::View<const T> x){
        if(x[0]!=1)return false;
        bool gray=false,free_rgb=false,nonlinear=false;
        for(int l=0;l<p.layers;++l){gray|=m[l].gray!=0;free_rgb|=!m[l].gray&&!(p.refinement&&p.smooth&&l==0);
            if(l){nonlinear|=m[l].mode!=0;
                if(m[l].X!=1||m[l].Y!=1||m[l].Z!=1||m[l].mode<0||m[l].mode>12)return false;}}
        return gray&&free_rgb&&nonlinear;
    }
    template<class Model>HD static bool run(const Model* m,general::Problem p,const T* target,
        general::View<T> x,general::View<T> prefix,general::View<const T> jacobian){
        int L=p.layers,N=4*L;general::View<const T> cx{x.data,x.stride};
        if(!eligible(m,p,cx))return false;
        general::Program<T,general::ZeroPrior,Model>::forward(m,p,cx,prefix);
        for(int r=0;r<3;++r){
            int anchor=-1;
            for(int l=L-1;l>=0;--l){if(m[l].gray||(p.refinement&&p.smooth&&l==0))continue;
                T q=std::abs(jacobian[r*N+L+3*l+r]);
                if(q>T(1e-10)){anchor=l;break;}}
            if(anchor<0)return false;
            T desired=target[r];
            for(int l=L-1;l>anchor;--l){T previous;
                if(!destination(m[l].mode,x[l],x[L+3*l+r],desired,prefix[4*(l-1)+r],previous))return false;
                desired=previous;
            }
            T color;
            if(anchor==0)color=desired;
            else{
                if(!source(m[anchor].mode,x[anchor],prefix[4*(anchor-1)+r],desired,x[L+3*anchor+r],color))return false;
            }
            if(!(color>=0&&color<=1&&configured::isfinite(color)))return false;
            x[L+3*anchor+r]=color;
        }
        return true;
    }
};
}
