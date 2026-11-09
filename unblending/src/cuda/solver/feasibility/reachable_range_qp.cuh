#pragma once
#include "reachable_ranges.cuh"
#include "jacobian_correction.cuh"

namespace range_refinement {
template<class T,class Model>HD T gaussian_energy(const Model* models,general::Problem p,general::View<T> x){T result=0;for(int l=0;l<p.layers;++l){if(x[l]==0)continue;T color[3];for(int r=0;r<3;++r)color[r]=x[p.layers+3*l+r];result+=x[l]*general::GaussianPrior::evaluate(models[l],color,static_cast<T*>(nullptr));}return result;}
// At opaque source-over, every prefix alpha is exactly one. Color values
// and color Jacobians need no repeated RGBA normalization or division by one.
// The general branch remains for other composition/background configurations.
template<class T,class Model>HD void forward_color_jacobian(const Model* models,general::Problem p,
    general::View<const T> x,general::View<T> prefix,general::View<T> jac){
    using P=general::Program<T,general::GaussianPrior,Model>;
    if(!opaque_source_over(models,p)||x[0]!=1){P::forward(models,p,x,prefix);general::composition_jacobian(models,p,x,{prefix.data,prefix.stride},jac);return;}
    int L=p.layers;for(int r=0;r<3;++r)prefix[r]=x[L+r];prefix[3]=1;
    for(int l=1;l<L;++l){T a=x[l];prefix[4*l+3]=1;for(int r=0;r<3;++r){T d=prefix[4*(l-1)+r];prefix[4*l+r]=a==0?d:a*general::blend(x[L+3*l+r],d,models[l].mode)+(1-a)*d;}}
    T adjoint[3]={1,1,1};
    for(int l=L-1;l>0;--l){T a=x[l];for(int r=0;r<3;++r){if(a==0){jac[L+3*l+r]=0;continue;}T s=x[L+3*l+r],d=prefix[4*(l-1)+r];jac[L+3*l+r]=adjoint[r]*a*general::grad_s(s,d,models[l].mode);adjoint[r]*=a*general::grad_d(s,d,models[l].mode)+(1-a);}}
    for(int r=0;r<3;++r)jac[L+r]=adjoint[r];
}
// Three-RGB Schur complement of the exact Gaussian quadratic under the
// current linearized constraint. This step alone is NOT a nonlinear solve.
template<class T,class Model>HD bool proposal_views(const Model* models,general::Problem p,general::View<const T> target,general::View<T> point,general::View<T> work){
    int L=p.layers;auto view=[&](int offset){return general::View<T>{work.data+size_t(offset)*work.stride,work.stride};};auto cv=[](general::View<T> x){return general::View<const T>{x.data,x.stride};};auto prefix=view(10*L),jac=view(14*L),next=view(18*L);
    for(int k=0;k<4*L;++k)next[k]=point[k];forward_color_jacobian(models,p,cv(point),prefix,jac);
    T rhs[3],K[6]{},dual[3];for(int r=0;r<3;++r){T wanted=target[r],low=work[3*(L-1)+r],high=work[3*L+3*(L-1)+r];wanted=wanted<low?low:(wanted>high?high:wanted);rhs[r]=prefix[4*(L-1)+r]-wanted;}
    bool degenerate=false;
    for(int l=0;l<L;++l){if(p.smooth&&l==0)continue;T a=point[l],j[3];for(int r=0;r<3;++r)j[r]=jac[L+3*l+r];if(a==0){if(j[0]!=0||j[1]!=0||j[2]!=0)degenerate=true;continue;}
        // Divide J by 2a BEFORE multiplying. Forming 1/(2a) can
        // overflow for subnormal a even when J/a and the Schur term are finite.
        T normalized[3];for(int r=0;r<3;++r){normalized[r]=j[r]/(T(2)*a);jac[L+3*l+r]=normalized[r];rhs[r]+=j[r]*(T(models[l].mean[r])-point[L+3*l+r]);}const int rows[6]={0,0,0,1,1,2},cols[6]={0,1,2,1,2,2};for(int k=0;k<6;++k)K[k]+=j[rows[k]]*T(models[l].covariance[3*rows[k]+cols[k]])*normalized[cols[k]];}
    T trace=K[0]+K[3]+K[5];if(degenerate||!normal_refinement::solve3(K,trace*T(sizeof(T)==4?1e-6:1e-12),rhs,dual))return false;
    for(int l=0;l<L;++l){if((p.smooth&&l==0)||point[l]==0)continue;for(int r=0;r<3;++r){T predicted=T(models[l].mean[r]);for(int s=0;s<3;++s)predicted-=T(models[l].covariance[3*r+s])*jac[L+3*l+s]*dual[s];next[L+3*l+r]=configured::clip(point[L+3*l+r]+T(.5)*(predicted-point[L+3*l+r]));}}
    return true;
}
template<class T,class Model>HD bool proposal(const Model* models,general::Problem p,int n,int index,const T* target,T* output,T* work){
    return proposal_views(models,p,general::View<const T>{target+index,size_t(n)},general::View<T>{output+index,size_t(n)},general::View<T>{work+index,size_t(n)});
}
template<class T,class Model>HD int improve_views(const Model* models,general::Problem p,general::View<const T> initial,general::View<const T> target,general::View<const T> alpha,general::View<const T> bg,general::View<T> point,general::View<T> work,T tolerance,T anchor,int rounds,int& accepted){
    accepted=0;int status=pixel_views(models,p,initial,target,alpha,bg,point,work,tolerance,anchor);if(status!=0&&status!=5)return status;
    int L=p.layers;auto trial=general::View<T>{work.data+size_t(22*L)*work.stride,work.stride};T best=gaussian_energy(models,p,point);if(!std::isfinite(best))return 2;
    for(int iteration=0;iteration<rounds;++iteration){if(!proposal_views(models,p,target,point,work))break;int next=pixel_views(models,p,general::View<const T>{work.data+size_t(18*L)*work.stride,work.stride},target,alpha,bg,trial,work,tolerance,T(0),true);T cost=gaussian_energy(models,p,trial);if((next==0||next==5)&&std::isfinite(cost)&&cost<best){for(int k=0;k<4*L;++k)point[k]=trial[k];best=cost;status=next;++accepted;}else break;}
    return status;
}
template<class T,class Model>HD int improve(const Model* models,general::Problem p,int n,int index,const T* initial,const T* target,const T* alpha,const T* bg,T* output,T* work,T tolerance,T anchor,int rounds,int& accepted){
    return improve_views(models,p,general::View<const T>{initial+index,size_t(n)},general::View<const T>{target+index,size_t(n)},
        general::View<const T>{alpha+index,size_t(n)},general::View<const T>{bg+index,size_t(n)},
        general::View<T>{output+index,size_t(n)},general::View<T>{work+index,size_t(n)},tolerance,anchor,rounds,accepted);
}
// Persistent slots reuse a bounded state tile across many image pixels. Only
// the slot's owner accesses its state. Image planes remain coalesced, unaltered.
template<class T,class Model,bool Improved>__global__ void tiled_kernel(const Model* model,general::Problem p,int n,const T* initial,const T* target,const T* alpha,const T* bg,T* output,T* scratch,T* ranges,int* status,T tolerance,T anchor,int rounds,int* accepted){
    int slot=blockIdx.x*blockDim.x+threadIdx.x,slots=gridDim.x*blockDim.x;
    auto work=general::View<T>{scratch+slot,size_t(slots)};
    for(int i=slot;i<n;i+=slots){
        auto cv=[&](const T* data){return general::View<const T>{data+i,size_t(n)};};auto point=general::View<T>{output+i,size_t(n)};
        // An unsupported prefix can return before writing the final ranges.
        for(int r=0;r<3;++r){work[3*(p.layers-1)+r]=0;work[3*p.layers+3*(p.layers-1)+r]=0;}
        if constexpr(Improved)status[i]=improve_views(model,p,cv(initial),cv(target),cv(alpha),cv(bg),point,work,tolerance,anchor,rounds,accepted[i]);
        else{status[i]=pixel_views(model,p,cv(initial),cv(target),cv(alpha),cv(bg),point,work,tolerance,anchor);accepted[i]=0;}
        for(int r=0;r<3;++r){ranges[size_t(r)*n+i]=work[3*(p.layers-1)+r];ranges[size_t(3+r)*n+i]=work[3*p.layers+3*(p.layers-1)+r];}
    }
}
template<class T,class Model>__global__ void plain_kernel(const Model* model,general::Problem p,int n,const T* initial,const T* target,const T* alpha,const T* bg,T* output,T* scratch,int* status,T tolerance,T anchor,int* accepted=nullptr){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){status[i]=pixel(model,p,n,i,initial,target,alpha,bg,output,scratch,tolerance,anchor);if(accepted)accepted[i]=0;}}
template<class T,class Model>__global__ void improved_kernel(const Model* model,general::Problem p,int n,const T* initial,const T* target,const T* alpha,const T* bg,T* output,T* scratch,int* status,T tolerance,T anchor,int rounds,int* accepted){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)status[i]=improve(model,p,n,i,initial,target,alpha,bg,output,scratch,tolerance,anchor,rounds,accepted[i]);}
// Split GPU form: smaller live register state per logical step. All arrays
// remain on device. Host only queues a fixed number of launches, no decisions.
template<class T,class Model>__global__ void proposal_kernel(const Model* model,general::Problem p,int n,const T* target,T* output,T* work,const int* status,int* active){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&active[i]&&(status[i]==0||status[i]==5))active[i]=proposal(model,p,n,i,target,output,work);}
template<class T,class Model>__global__ void trial_kernel(const Model* model,general::Problem p,int n,const T* target,const T* alpha,const T* bg,T* work,const int* active,int* trial_status,T tolerance){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&active[i])trial_status[i]=pixel(model,p,n,i,work+size_t(18*p.layers)*n,target,alpha,bg,work+size_t(22*p.layers)*n,work,tolerance,T(0),true);}
template<class T,class Model>__global__ void select_kernel(const Model* model,general::Problem p,int n,T* output,T* work,int* status,const int* trial_status,int* active,int* accepted){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||!active[i])return;auto point=general::View<T>{output+i,size_t(n)},trial=general::View<T>{work+size_t(22*p.layers)*n+i,size_t(n)};int next=trial_status[i];T old=gaussian_energy(model,p,point),cost=gaussian_energy(model,p,trial);if((next==0||next==5)&&std::isfinite(cost)&&cost<old){for(int k=0;k<4*p.layers;++k)point[k]=trial[k];status[i]=next;++accepted[i];}else active[i]=0;}
__global__ void active_kernel(int n,const int* status,int* active){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)active[i]=status[i]==0||status[i]==5;}
}
