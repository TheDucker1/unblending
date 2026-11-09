#pragma once
#include "trace.cuh"
#include "optimizer.cuh"
#include "range_union.cuh"
#include "two_layer_gray.cuh"
#include "sqp_math.cuh"

namespace general {
#ifdef __CUDACC__
#define RESTORE_OUTLINED __noinline__
#else
#define RESTORE_OUTLINED
#endif
struct ZeroPrior {
    template<class T,class Model>HD static T evaluate(const Model&,const T*,T* gradient){if(gradient)for(int r=0;r<3;++r)gradient[r]=0;return 0;}
};

// Feasibility restoration has a different job from energy minimization. Its
// 3/4-dimensional Gram system includes RGB derivatives w.r.t. BOTH alpha and
// color, and exact gray tangent directions. It can repair an infeasible AL
// incumbent without imposing an energy budget relative to that invalid point.
template<class T,class Model>struct ConstraintRestore {
    using P=Program<T,ZeroPrior,Model>;
    HD static bool identical(T a,T b){return a==b&&(a!=T(0)||std::signbit(a)==std::signbit(b));}
    HD static RESTORE_OUTLINED T norm(const Model* m,Problem p,const T* target,View<const T> alpha,View<const T> x){
        probe::hit(probe::Norm);int L=p.layers;T rgb[3],da=x[0],result=0;for(int k=0;k<4*L;++k)if(!std::isfinite(x[k])||x[k]<0||x[k]>1)return INFINITY;if(p.opaque&&x[0]!=1)return INFINITY;
        for(int r=0;r<3;++r)rgb[r]=x[L+r];for(int l=1;l<L;++l){T a=x[l],next=T(m[l].X)*a*da+T(m[l].Y)*a*(1-da)+T(m[l].Z)*da*(1-a);for(int r=0;r<3;++r)rgb[r]=range_union::native_color(m[l],a,da,next,x[L+3*l+r],rgb[r]);da=next;}
        for(int r=0;r<3;++r){T d=rgb[r]-target[r];result+=d*d;}if(p.refinement)for(int l=0;l<L;++l){T d=x[l]-alpha[l];result+=d*d;}else{T d=da-1;result+=d*d;}
        for(int l=0;l<L;++l)if(m[l].gray){T length=0;for(int r=0;r<3;++r)length+=x[L+3*l+r]*x[L+3*l+r];length=configured::root(length);for(int r=0;r<3;++r){T d=T(1.7320508075688772935)*x[L+3*l+r]-length;result+=d*d;}}
        return std::isfinite(result)?result:T(INFINITY);
    }
    HD static bool solve(int count,const T* gram,const T* rhs,T* result){
        if(count==3&&sqp_math::spd3(gram,rhs,result))return true;T a[20]{},trace=0;for(int r=0;r<count;++r)trace+=gram[4*r+r];if(!(trace>0&&std::isfinite(trace)))return false;
        for(int r=0;r<count;++r){result[r]=0;for(int s=0;s<count;++s)a[5*r+s]=gram[4*r+s]+(r==s?trace*T(sizeof(T)==4?1e-6:1e-12):T(0));a[5*r+count]=rhs[r];}
        for(int k=0;k<count;++k){int pivot=k;for(int r=k+1;r<count;++r)if(std::abs(a[5*r+k])>std::abs(a[5*pivot+k]))pivot=r;if(!(std::abs(a[5*pivot+k])>0))return false;if(pivot!=k)for(int s=k;s<=count;++s){T v=a[5*k+s];a[5*k+s]=a[5*pivot+s];a[5*pivot+s]=v;}
            for(int r=k+1;r<count;++r){T scale=a[5*r+k]/a[5*k+k];for(int s=k;s<=count;++s)a[5*r+s]-=scale*a[5*k+s];}}
        for(int r=count-1;r>=0;--r){T v=a[5*r+count];for(int s=r+1;s<count;++s)v-=a[5*r+s]*result[s];result[r]=v/a[5*r+r];if(!std::isfinite(result[r]))return false;}return true;
    }
    HD static void project(const Model* m,Problem p,View<const T> alpha,const T* bg,View<T> x){
        for(int k=0;k<4*p.layers;++k){T low=Optimizer<T,ZeroPrior,Model>::bound(p,alpha,bg,k,false),high=Optimizer<T,ZeroPrior,Model>::bound(p,alpha,bg,k,true);T v=std::isfinite(x[k])?x[k]:T(.5);x[k]=v<low?low:(v>high?high:v);}
        if(p.refinement)for(int l=0;l<p.layers;++l)x[l]=p.opaque&&l==0?T(1):alpha[l];Optimizer<T,ZeroPrior,Model>::project_gray(m,p,x);
    }
    HD static RESTORE_OUTLINED bool neutral(const Model* m,Problem p,const T* target,View<T> x,int variant){
        if(p.refinement)return false;for(int r=0;r<3;++r)if(target[r]<0||target[r]>1)return false;int L=p.layers;bool so=true,plus=true;for(int l=1;l<L;++l){so&=m[l].X==1&&m[l].Y==1&&m[l].Z==1;plus&=m[l].X==2&&m[l].Y==1&&m[l].Z==1;}
        if(!so&&!plus){
            // General source-only witness, independent of every blend mode:
            // zero prefix opacity makes the last output A=Y*a and preRGB=Y*a*s.
            // With a=1/Y, s=target, both original constraints hold. A free
            // background and nongray last layer are required. Optimize the
            // actual prior from this feasible seed afterward.
            if(p.opaque||L<2||m[L-1].Y<1||m[L-1].gray)return false;
            for(int l=0;l<L;++l){x[l]=0;for(int r=0;r<3;++r)x[L+3*l+r]=configured::clip(T(m[l].mean[r]));}
            Optimizer<T,ZeroPrior,Model>::project_gray(m,p,x);x[L-1]=T(1)/T(m[L-1].Y);
            for(int r=0;r<3;++r)x[L+3*(L-1)+r]=target[r];return true;
        }
        int layer=-1;for(int l=0;l<L;++l){x[l]=0;for(int r=0;r<3;++r)x[L+3*l+r]=configured::clip(T(m[l].mean[r]));if(!m[l].gray&&layer<0)layer=l;}Optimizer<T,ZeroPrior,Model>::project_gray(m,p,x);
        if(layer<0){x[0]=1;for(int r=0;r<3;++r)x[L+r]=(target[0]+target[1]+target[2])/T(3);return true;}
        if(plus){if(p.opaque&&layer!=0)return false;x[layer]=1;for(int r=0;r<3;++r)x[L+3*layer+r]=target[r];return true;}
        x[0]=1;if(layer==0){for(int r=0;r<3;++r)x[L+r]=target[r];return true;}x[layer]=1;int mode=m[layer].mode;T wanted[3];for(int r=0;r<3;++r)wanted[r]=target[r];
        if(mode==9&&variant>0){int above=-1;for(int l=layer+1;l<L;++l)if(m[l].gray&&m[l].mode==9){above=l;break;}if(above<0)return false;x[above]=1;for(int r=0;r<3;++r){x[L+3*above+r]=variant==1?T(0):T(1);wanted[r]=variant==1?configured::root(target[r]):range_refinement::soft_inverse1(target[r],T(1),T(0),T(0));}}
        T minimum=wanted[0],maximum=wanted[0];for(int r=1;r<3;++r){if(wanted[r]<minimum)minimum=wanted[r];if(wanted[r]>maximum)maximum=wanted[r];}T d=T(.5);
        if(mode==1||mode==4)d=1;else if(mode==2||mode==5||mode==10||mode==11||mode==12)d=0;
        else if(mode==6){d=T(.5)*minimum;if(maximum==0)d=0;else if(d<range_union::epsilon<T>())d=range_union::epsilon<T>();for(int r=0;r<3;++r)if(wanted[r]<d)wanted[r]=d;}
        else if(mode==7){d=T(.5)*(1+maximum);T cut=std::nextafter(range_union::cutoff<T>(7),-T(INFINITY));if(minimum==1)d=1;else if(d>cut)d=cut;for(int r=0;r<3;++r)if(wanted[r]>d)wanted[r]=d;}
        else if(mode==9){T low=range_refinement::soft_inverse1(maximum,T(1),T(0),T(0)),high=configured::root(minimum);if(low>high)return false;d=d<low?low:(d>high?high:d);}
        for(int r=0;r<3;++r){x[L+r]=d;T source;auto set=range_union::sources(range_refinement::Nearest<T,2>{x[L+3*layer+r]},mode,d,wanted[r],T(1),T(0),T(0));if(!set.nearest(T(.5),source))return false;x[L+3*layer+r]=source;}return true;
    }
    // Three-layer gray/RGB/gray Dodge chain. With middle alpha a, top alpha u,
    // background b and top multiplier h=1/(1-s_top), its RGB interval is
    // D_h([b,a+(1-a)b]), D_h(d)=u*min(h*d,1)+(1-u)*d.
    // Set its lower endpoint to min(target), choose the smallest h making its
    // upper endpoint reach max(target), then invert each channel exactly.
    // Burn follows by complementing all colors. Native evaluation validates
    // boundary roundoff afterward; this is a seed, not a new constraint model.
    HD static RESTORE_OUTLINED bool gray_three_seed(const Model* m,Problem p,const T* target,View<const T> alpha,const T* bg,View<T> x){
        if(p.layers!=3||!p.refinement||alpha[0]!=1||!m[0].gray||m[1].gray||!m[2].gray||m[1].mode!=m[2].mode||(m[1].mode!=6&&m[1].mode!=7))return false;
        for(int l=1;l<3;++l)if(m[l].X!=1||m[l].Y!=1||m[l].Z!=1)return false;
        bool burn=m[1].mode==7;T t[3],low=1,high=0,a=alpha[1],u=alpha[2],eps=T(64)*std::numeric_limits<T>::epsilon();
        if(!(a>eps&&u>eps&&a<=1&&u<=1))return false;
        for(int r=0;r<3;++r){t[r]=burn?T(1)-target[r]:target[r];if(t[r]<0||t[r]>1)return false;if(t[r]<low)low=t[r];if(t[r]>high)high=t[r];}
        T b,h,c=1-u,d=1-a;
        if(p.smooth){b=burn?T(1)-bg[0]:bg[0];T upper=a+d*b;if(!(upper>0))return false;h=(high-c*upper)/(u*upper);if(h<1)h=1;}
        else{T k=(high-d*low)/a;if(k<1)k=1;b=low/k;h=(k-c)/u;}
        if(!(b>=T(2)*range_union::epsilon<T>()&&b<=1&&h>=1&&std::isfinite(h)))return false;
        auto top=[&](T value){T v=h*value;return u*(v<1?v:T(1))+c*value;};T upper=a+d*b;
        if(top(b)>low+eps||top(upper)<high-eps)return false;
        T k=c+u*h,split=u+c/h,top_source=1-T(1)/h;
        for(int l=0;l<3;++l)x[l]=alpha[l];
        for(int r=0;r<3;++r){T middle=t[r]<=split?t[r]/k:(c>0?(t[r]-u)/c:upper);T blend_value=(middle-d*b)/a;
            if(blend_value<b-eps||blend_value>1+eps)return false;
            if(blend_value<b)blend_value=b;if(blend_value>1)blend_value=1;
            T source=blend_value>0?1-b/blend_value:T(0);source=configured::clip(source);
            x[3+r]=p.smooth?bg[r]:(burn?T(1)-b:b);x[6+r]=burn?T(1)-source:source;x[9+r]=burn?T(1)-top_source:top_source;
        }
        return true;
    }

    // Jump out of bound-induced local minima with an explicit native witness.
    // Eligibility depends on the configured graph and the fixed-input contract.
    HD static RESTORE_OUTLINED bool identity_refinement_seed(const Model* m,Problem p,const T* target,View<const T> alpha,View<T> x){
        if(!p.refinement||p.smooth||m[0].gray||alpha[0]!=1)return false;
        for(int r=0;r<3;++r)if(target[r]<0||target[r]>1)return false;
        int L=p.layers;
        for(int l=1;l<L;++l){
            if(m[l].X!=1||m[l].Y!=1||m[l].Z!=1||m[l].mode<0||m[l].mode>12)return false;
            if(m[l].mode==0&&m[l].gray&&alpha[l]!=0&&(target[0]!=target[1]||target[1]!=target[2]))return false;
        }
        // f(identity,d)=d: Multiply/Darken/Burn use1, Overlay/HardLight/
        // SoftLight use.5, Screen/Lighten/Dodge/Difference/Exclusion/Add use0.
        // Normal uses source=target when RGB is free. With an opaque prefix,
        // a*f(identity,d)+(1-a)*d retains d at ANY prescribed source opacity.
        for(int l=0;l<L;++l){int mode=m[l].mode;T neutral=(mode==1||mode==4||mode==7)?T(1):((mode==3||mode==8||mode==9)?T(.5):T(0));
            x[l]=alpha[l];for(int r=0;r<3;++r)x[L+3*l+r]=l==0||mode==0?target[r]:neutral;
            if(l&&m[l].mode==0&&m[l].gray&&alpha[l]==0)for(int r=0;r<3;++r)x[L+3*l+r]=0;}
        // Dodge/Burn's strict epsilon branches can perturb the ideal identity.
        // run() always evaluates the native residual before claiming feasibility.
        return true;
    }
    HD static RESTORE_OUTLINED bool refinement_seed(const Model* m,Problem p,const T* target,View<const T> alpha,const T* bg,View<T> x,int variant){
        if(variant==0&&gray_three_seed(m,p,target,alpha,bg,x))return true;
        if(variant==0&&identity_refinement_seed(m,p,target,alpha,x))return true;
        if(!p.refinement||alpha[0]!=1)return false;bool burn=false,dodge=false;
        for(int l=1;l<p.layers;++l){if(m[l].X!=1||m[l].Y!=1||m[l].Z!=1||(m[l].mode!=6&&m[l].mode!=7))return false;burn|=m[l].mode==7;dodge|=m[l].mode==6;}
        if(!burn&&!dodge)return false;
        T low=target[0],high=target[0];for(int r=0;r<3;++r){if(target[r]<0||target[r]>1)return false;if(target[r]<low)low=target[r];if(target[r]>high)high=target[r];}
        for(int l=0;l<p.layers;++l)x[l]=alpha[l];
        for(int r=0;r<3;++r){T value=m[0].gray?(burn&&!dodge?high:(dodge&&!burn?low:(low+high)/2)):target[r];
            if(variant==1)value=burn&&!dodge?(value+1)/2:(dodge&&!burn?value/2:value);
            if(variant==2)value=burn&&!dodge?T(.95):(dodge&&!burn?T(.05):T(.5));
            x[p.layers+r]=p.smooth?bg[r]:value;
        }
        for(int l=1;l<p.layers;++l)for(int r=0;r<3;++r)x[p.layers+3*l+r]=m[l].mode==7?(variant==2?T(1)-epsilon_value():T(1)):(variant==2?epsilon_value():T(0));
        return true;
    }
    HD static T epsilon_value(){return range_union::epsilon<T>();}
    HD static RESTORE_OUTLINED bool run(const Model* model,Problem p,const T* target,View<const T> alpha,const T* bg,View<const T> initial,View<T> output,View<T> memory,T tolerance,int iterations=48,int seeds=5){
        if(p.refinement&&p.smooth&&model[0].gray&&(bg[0]!=bg[1]||bg[1]!=bg[2]))return false;
        probe::hit(probe::RestoreCall);int L=p.layers,N=4*L,C=3+L;auto at=[&](int offset){return View<T>{memory.data+size_t(offset)*memory.stride,memory.stride};};auto cv=[](View<T> v){return View<const T>{v.data,v.stride};};auto x=at(0),trial=at(N),prefix=at(2*N),lambda=at(7*N),residual=at(7*N+C);
        T incumbent=norm(model,p,target,alpha,cv(output)),best=incumbent,goal=tolerance*tolerance*T(.0025);bool changed=false;
        // Exact geometry is preferable to a local iteration trapped on a flat
        // gray-source branch. The seed is validated against the original native
        // equations; actual-prior minimization still follows restoration.
        if(p.layers==2&&best>goal&&TwoLayerGraySeed<T,Model>::run(model,p,target,alpha,x,tolerance)){
            best=norm(model,p,target,alpha,cv(x));if(best<incumbent){for(int k=0;k<N;++k)output[k]=x[k];changed=true;}
        }
        #pragma unroll 1
        bool duplicate_initial=false,duplicate_neutral=false;
        if(seeds>1){for(int k=0;k<N;++k)x[k]=initial[k];project(model,p,alpha,bg,x);duplicate_initial=true;for(int k=0;k<N;++k)duplicate_initial&=identical(x[k],output[k]);}
        if(!p.refinement){int first=-1;for(int l=0;l<L;++l)if(!model[l].gray){first=l;break;}duplicate_neutral=first<1||model[first].mode!=9;}
        for(int seed=0;seed<seeds&&best>goal;++seed){if((seed==1&&duplicate_initial)||(seed>=3&&duplicate_neutral))continue;probe::hit(probe::RestoreSeed);for(int k=0;k<N;++k)x[k]=seed==0?output[k]:initial[k];if(seed>=2&&!(p.refinement?refinement_seed(model,p,target,alpha,bg,x,seed-2):neutral(model,p,target,x,seed-2)))continue;project(model,p,alpha,bg,x);T value=norm(model,p,target,alpha,cv(x));
            #pragma unroll 1
            for(int iteration=0;iteration<iterations;++iteration){if(value<best){best=value;changed=true;for(int k=0;k<N;++k)output[k]=x[k];}if(value<=goal)break;probe::hit(probe::RestoreIteration);
                P::forward(model,p,cv(x),prefix);auto J=at(3*N);sqp_math::jacobian(model,p,cv(x),cv(prefix),J);
                T rhs[4];for(int r=0;r<3;++r)rhs[r]=prefix[4*(L-1)+r]-target[r];rhs[3]=prefix[4*(L-1)+3]-1;int rows=!p.refinement&&std::abs(rhs[3])>T(1e-10)?4:3;
                auto free=[&](int k,T direction){if(k<L&&(p.refinement||(p.opaque&&k==0)))return false;if(p.refinement&&p.smooth&&k>=L&&k<L+3)return false;return !((x[k]<=0&&direction<0)||(x[k]>=1&&direction>0));};
                T gram[16]{},dual[4]{};
                // Color columns are diagonal across RGB. A gray triple is one
                // shared direction: J_gray*J_gray^T/3 in the Euclidean metric.
                for(int l=0;l<L;++l){
                    if(!p.refinement&&!(p.opaque&&l==0)){
                        T steepest=0;for(int row=0;row<rows;++row)steepest-=J[row*N+l]*rhs[row];
                        if(free(l,steepest))for(int r=0;r<rows;++r)for(int s=0;s<rows;++s)gram[4*r+s]+=J[r*N+l]*J[s*N+l];
                    }
                    if(p.refinement&&p.smooth&&l==0)continue;
                    T j[3];for(int r=0;r<3;++r)j[r]=J[r*N+L+3*l+r];
                    if(model[l].gray){T steepest=0;for(int r=0;r<3;++r)steepest-=j[r]*rhs[r]/T(3);
                        if(free(L+3*l,steepest))for(int r=0;r<3;++r)for(int s=0;s<3;++s)gram[4*r+s]+=j[r]*j[s]/T(3);
                    }else for(int r=0;r<3;++r)if(free(L+3*l+r,-j[r]*rhs[r]))gram[4*r+r]+=j[r]*j[r];
                }
                auto previous=at(7*N);bool previous_valid=false;bool solved=solve(rows,gram,rhs,dual),moved=false;
                for(int method=0;method<2&&!moved;++method){if(method==0&&!solved)continue;T amount=1;if(method){probe::hit(probe::GradientFallback);T trace=0;for(int r=0;r<rows;++r)trace+=gram[4*r+r];if(!(trace>0))continue;amount=1/trace;}
                    #pragma unroll 1
                    for(int backtrack=0;backtrack<16;++backtrack){probe::hit(probe::RestoreTrial);bool any_change=false;
                        for(int l=0;l<L;++l){
                            T direction=0;for(int row=0;row<rows;++row)direction-=J[row*N+l]*(method?rhs[row]:dual[row]);trial[l]=free(l,direction)?configured::clip(x[l]+amount*direction):x[l];any_change|=!identical(trial[l],x[l]);
                            T gray=0;if(model[l].gray)for(int r=0;r<3;++r)gray-=J[r*N+L+3*l+r]*(method?rhs[r]:dual[r])/T(3);
                            for(int r=0;r<3;++r){int k=L+3*l+r;T move=model[l].gray?gray:-J[r*N+k]*(method?rhs[r]:dual[r]);trial[k]=free(k,move)?configured::clip(x[k]+amount*move):x[k];any_change|=!identical(trial[k],x[k]);}
                        }
                        if(!any_change)break;project(model,p,alpha,bg,trial);bool cached=previous_valid;
                        if(cached)for(int k=0;k<N;++k)cached&=identical(previous[k],trial[k]);if(cached){amount*=T(.5);continue;}
                        T next=norm(model,p,target,alpha,cv(trial));if(next<value){for(int k=0;k<N;++k)x[k]=trial[k];value=next;moved=true;break;}for(int k=0;k<N;++k)previous[k]=trial[k];previous_valid=true;amount*=T(.5);
                    }}
                if(!moved){probe::hit(probe::RestoreStalled);break;}if(iteration==iterations-1)probe::hit(probe::RestoreIterationLimit);
            }
            if(value<best){best=value;changed=true;for(int k=0;k<N;++k)output[k]=x[k];}
        }
        probe::end_restore();return changed&&best<incumbent;
    }
};
#undef RESTORE_OUTLINED
}
