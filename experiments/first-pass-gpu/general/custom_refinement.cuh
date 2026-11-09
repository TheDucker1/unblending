#pragma once
#include "range_union.cuh"
#include "custom_model.hpp"
#include "coupled_sqp.cuh"
#include "cube_retract.cuh"

namespace custom_refinement {
using general::View;
using Prior=general::PolynomialPrior;
using Model=custom_backend::Model;
// Topology is shared by all pixels. Select the ordinary kernel on the host;
// numeric parameters and layer counts remain runtime data in reusable kernels.
struct Graph {
    bool known=false,gray=false,source_over=true,plus=true,cube=true,gaussian=true;
    Graph(const Model* models,int L){
        known=models!=nullptr;if(!known)return;
        for(int l=0;l<L;++l){gray|=models[l].gray!=0;gaussian&=models[l].term_count==0;if(l){source_over&=models[l].X==1&&models[l].Y==1&&models[l].Z==1;plus&=models[l].X==2&&models[l].Y==1&&models[l].Z==1;cube&=models[l].mode>=0&&models[l].mode<=11&&models[l].mode!=6&&models[l].mode!=7;}}
    }
    // 0 preserves the original combined path for callers without a host plan.
    int path(general::Problem p,bool inverse)const{
        if(!known)return 0;
        if(cube&&source_over&&!gray&&p.opaque){
            bool specialized=gaussian&&(p.layers==2||p.layers==3||p.layers==5||p.layers==7);
            if(!p.refinement)return specialized?5:4;
            if(p.fixed_alpha&&!p.smooth)return specialized?7:6;
        }
        return !inverse||gray||(!p.refinement&&(!(source_over&&p.opaque)&&!plus))?2:1;
    }
};
HD inline size_t inverse_words(int layers){return range_union::words(layers)+size_t(10)*layers;}
HD inline size_t words(int layers,bool inverse=true){size_t a=inverse?inverse_words(layers):0,coupled=size_t(76)*layers;return a>coupled?a:coupled;}
// Coalesce each variable across a warp, while keeping that warp's iterative
// state in one compact region instead of spacing its variables by image width.
template<class T>HD View<T> state_view(T* memory,int slot,int capacity,size_t count,int tile){
    if(tile==0)return {memory+slot,size_t(capacity)};
    int begin=(slot/tile)*tile,width=capacity-begin;if(width>tile)width=tile;
    return {memory+size_t(begin)*count+(slot-begin),size_t(width)};
}

template<class T>HD T local_metric(const Model& m,const T* color,T* gradient,T* covariance,T* gray_inverse){
    T value=general::GaussianPrior::evaluate(m,color,gradient);const int row[6]={0,0,0,1,1,2},col[6]={0,1,2,1,2,2};
    // With no additional terms, H=S+S^T is constant. Input::metric already
    // computes 2*H^-1 and 1/(ones^T H ones) once per model, in double precision.
    // Reuse them across pixels and rounds on BOTH devices. The objective and
    // its full-covariance gradient above are still evaluated at this color.
    if(m.term_count==0){for(int k=0;k<6;++k)covariance[k]=T(m.covariance[3*row[k]+col[k]]);if(gray_inverse)*gray_inverse=T(m.gray_inverse);return value;}
    T h[6];
    for(int k=0;k<6;++k)h[k]=T(m.inverse[3*row[k]+col[k]])+T(m.inverse[3*col[k]+row[k]]);
    for(int j=0;j<m.term_count;++j){const double* term=m.terms+5*j;T d=-T(term[3]);for(int r=0;r<3;++r)d+=T(term[r])*color[r];T square=d*d,first=T(4)*T(term[4])*square*d,second=T(12)*T(term[4])*square;
        value+=T(term[4])*square*square;for(int r=0;r<3;++r)gradient[r]+=first*T(term[r]);for(int k=0;k<6;++k)h[k]+=second*T(term[row[k]])*T(term[col[k]]);}
    T gray=h[0]+h[3]+h[5]+T(2)*(h[1]+h[2]+h[4]);
    T scale=h[0]>h[3]?h[0]:h[3];if(h[5]>scale)scale=h[5];bool valid=scale>0&&std::isfinite(scale);if(valid){for(int k=0;k<6;++k)h[k]/=scale;
        T cofactor[6]={h[3]*h[5]-h[4]*h[4],h[2]*h[4]-h[1]*h[5],h[1]*h[4]-h[2]*h[3],h[0]*h[5]-h[2]*h[2],h[1]*h[2]-h[0]*h[4],h[0]*h[3]-h[1]*h[1]};T determinant=h[0]*cofactor[0]+h[1]*cofactor[1]+h[2]*cofactor[2];
        valid=h[0]>0&&cofactor[5]>0&&determinant>0;T factor=valid?T(2)/(determinant*scale):T(0);for(int k=0;k<6;++k){covariance[k]=cofactor[k]*factor;valid=valid&&std::isfinite(covariance[k]);}}
    if(!valid)for(int k=0;k<6;++k)covariance[k]=T(m.covariance[3*row[k]+col[k]]);
    if(gray_inverse){T inverse=valid&&gray>0?T(1)/gray:T(m.gray_inverse);*gray_inverse=inverse>0&&std::isfinite(inverse)?inverse:T(1);}
    return value;
}

// The first-pass opacity trial needs d(E + multiplier.RGB)/d(alpha), not
// source-color derivatives, gray penalties or a full Lagrangian evaluation.
// Preconditions: first pass, no gray layers, and prefix freshly computed for x.
// For opaque SO, alpha_BG is fixed and all prefix alphas are one. Restrict the
// chain rule before differentiating: dc_l/d(alpha_l)=blend(s_l,d)-d, while the
// backward RGB adjoint is multiplied by a*blend_d+(1-a). General RGBA below
// retains normalization/alpha adjoints for Plus and other composition graphs.
template<class T>HD void alpha_adjoint(const Model* model,general::Problem p,View<const T> x,View<const T> prefix,const T* multiplier,View<T> gradient,bool opaque_so){
    int L=p.layers;T sum=0,square=0;
    for(int l=0;l<L;++l){T color[3];for(int r=0;r<3;++r)color[r]=x[L+3*l+r];gradient[l]=Prior::evaluate(model[l],color,static_cast<T*>(nullptr))+(p.minimum?T(.01):T(0));sum+=x[l];square+=x[l]*x[l];}
    if(p.sparsity)for(int l=0;l<L;++l)gradient[l]+=T(p.sparsity_weight)*(square-T(2)*x[l]*sum)/(square*square);
    T adj[3]={multiplier[0],multiplier[1],multiplier[2]},alpha_adj=0;
    for(int l=L-1;l>0;--l){const auto& m=model[l];T a=x[l],ga=0;
        if(opaque_so){
            for(int r=0;r<3;++r){T s=x[L+3*l+r],d=prefix[4*(l-1)+r],v=adj[r];ga+=v*(general::blend(s,d,m.mode)-d);adj[r]=v*(a*general::grad_d(s,d,m.mode)+(1-a));}
        }else{
            T d=prefix[4*(l-1)+3],A=prefix[4*l+3],da=T(m.X)*d+T(m.Y)*(1-d)-T(m.Z)*d,dd=T(m.X)*a-T(m.Y)*a+T(m.Z)*(1-a);
            ga=alpha_adj*da;T previous_alpha=alpha_adj*dd;
            for(int r=0;r<3;++r){T s=x[L+3*l+r],c=prefix[4*(l-1)+r],f=general::blend(s,c,m.mode),v=adj[r],out=prefix[4*l+r];
                T dc=a*d*general::grad_d(s,c,m.mode)+T(m.Z)*d*(1-a),pa=f*d+T(m.Y)*(1-d)*s-T(m.Z)*d*c,pd=f*a-T(m.Y)*a*s+T(m.Z)*(1-a)*c;
                ga+=v*(A>T(1e-12)?(pa-out*da)/A:pa);previous_alpha+=v*(A>T(1e-12)?(pd-out*dd)/A:pd);adj[r]=v*(A>T(1e-12)?dc/A:dc);
            }
            alpha_adj=previous_alpha;
        }
        gradient[l]+=ga;
    }
    if(!opaque_so)gradient[0]+=alpha_adj;
}

// Positive local block metric H_l (constant majorizer fallback), actual custom
// gradient g_l. Eliminating
// color increments from J*delta=-r leaves only the 3x3 system
// (sum J_l H_l^-1 J_l^T / alpha_l) lambda = r-sum J_l H_l^-1 gradD_l.
// covariance=2 H_l^-1 is cached as six values per layer; the 3x3 inverse is
// explicit and uses no optimizer history. Cross-channel terms survive both sides.
template<class T,bool Cube=false>HD bool proposal(const Model* model,general::Problem p,View<const T> target,View<T> point,View<T> work,T* multiplier=nullptr){
    int L=p.layers;auto view=[&](int offset){return View<T>{work.data+size_t(offset)*work.stride,work.stride};};auto cv=[](View<T> x){return View<const T>{x.data,x.stride};};
    auto prefix=view(10*L),jac=view(14*L),next=view(18*L);View<T> metric{work.data+((Cube?size_t(26)*L:range_union::words(L))+size_t(4)*L)*work.stride,work.stride};for(int k=0;k<4*L;++k)next[k]=point[k];range_refinement::forward_color_jacobian(model,p,cv(point),prefix,jac);
    T rhs[3],K[6]{},dual[3];for(int r=0;r<3;++r)rhs[r]=prefix[4*(L-1)+r]-target[r];bool degenerate=false;
    const int rows[6]={0,0,0,1,1,2},cols[6]={0,1,2,1,2,2};
    for(int l=0;l<L;++l){if(p.smooth&&l==0)continue;T a=point[l],j[3];for(int r=0;r<3;++r)j[r]=jac[L+3*l+r];if(a==0){if(j[0]!=0||j[1]!=0||j[2]!=0)degenerate=true;continue;}
        T color[3],gradient[3],covariance[6];for(int r=0;r<3;++r)color[r]=point[L+3*l+r];local_metric(model[l],color,gradient,covariance);for(int k=0;k<6;++k)metric[6*l+k]=covariance[k];const int indices[9]={0,1,2,1,3,4,2,4,5};
        T normalized[3];for(int r=0;r<3;++r){T move=0;for(int s=0;s<3;++s)move+=T(.5)*covariance[indices[3*r+s]]*gradient[s];next[L+3*l+r]=point[L+3*l+r]-move;rhs[r]-=j[r]*move;normalized[r]=j[r]/(T(2)*a);jac[L+3*l+r]=normalized[r];}
        for(int k=0;k<6;++k)K[k]+=j[rows[k]]*covariance[k]*normalized[cols[k]];
    }
    T trace=K[0]+K[3]+K[5];if(degenerate||!std::isfinite(trace)||trace<0)return false;
    if(trace>0){if(!normal_refinement::solve3(K,trace*T(sizeof(T)==4?1e-6:1e-12),rhs,dual))return false;}else for(int r=0;r<3;++r)dual[r]=0;
    if(multiplier)for(int r=0;r<3;++r)multiplier[r]=dual[r];
    const int indices[9]={0,1,2,1,3,4,2,4,5};for(int l=0;l<L;++l){if((p.smooth&&l==0)||point[l]==0)continue;for(int r=0;r<3;++r){T predicted=next[L+3*l+r];for(int s=0;s<3;++s)predicted-=metric[6*l+indices[3*r+s]]*jac[L+3*l+s]*dual[s];if(!std::isfinite(predicted))return false;next[L+3*l+r]=predicted;}}
    return true;
}

template<class T,bool Cube=false>HD int improve(const Model* model,general::Problem p,View<const T> initial,View<const T> target,View<const T> alpha,View<const T> bg,View<T> point,View<T> work,T tolerance,int rounds,int& accepted,int& values){
    int L=p.layers;accepted=values=0;T limit=sizeof(T)==4?T(32):T(INFINITY);int status;if constexpr(Cube)status=reachable_cube::pixel(model,p,initial,target,alpha,bg,point,work,tolerance,T(0),false,limit);else status=range_union::pixel(model,p,initial,target,alpha,bg,point,work,tolerance,T(0),false,limit);if(status!=0&&status!=5)return status;
    using P=general::Program<T,Prior,Model>;T best=P::energy(model,p,View<const T>{point.data,point.stride});++values;if(!std::isfinite(best))return 2;
    if(rounds==0)return status;T goal[3];for(int r=0;r<3;++r){if constexpr(Cube)goal[r]=configured::clip(target[r]);else{range_union::Prefix<T> last{work,L,L-1,r};if(!last.nearest(target[r],goal[r]))return 9;}}
    View<T> preferred{work.data+size_t(18*L)*work.stride,work.stride},trial{work.data+size_t(22*L)*work.stride,work.stride};
    for(int iteration=0;iteration<rounds;++iteration){if(!proposal<T,Cube>(model,p,View<const T>{goal,1},point,work))break;bool moved=false;
        for(int attempt=0;attempt<5;++attempt){for(int k=0;k<4*L;++k)preferred[k]=point[k]+T(.5)*(preferred[k]-point[k]);
            int code;if constexpr(Cube)code=reachable_cube::pixel(model,p,View<const T>{preferred.data,preferred.stride},target,alpha,bg,trial,work,tolerance,T(0),true,limit);else code=range_union::pixel(model,p,View<const T>{preferred.data,preferred.stride},target,alpha,bg,trial,work,tolerance,T(0),true,limit);T energy=P::energy(model,p,View<const T>{trial.data,trial.stride});++values;
            if((code==0||code==5)&&std::isfinite(energy)&&energy<best){for(int k=0;k<4*L;++k)point[k]=trial[k];best=energy;status=code;++accepted;moved=true;break;}}
        if(!moved)break;
    }
    return status;
}

// Fast first-pass domain matches the original complete pipeline's normalizers:
// opaque source-over or normalized Plus. RGB is re-solved after each opacity
// trial. The custom Lagrangian supplies the envelope-gradient estimate; actual
// custom energy and native constraints decide acceptance, never CPU layers.
template<class T,bool Cube=false>HD int first(const Model* model,general::Problem p,View<const T> initial,View<const T> target,View<const T> bg,View<T> point,View<T> work,T tolerance,int rounds,int& accepted,int& values){
    accepted=values=0;int L=p.layers;bool so=true,plus=true;for(int l=0;l<L;++l){if(model[l].gray)return 8;if(l){so&=model[l].X==1&&model[l].Y==1&&model[l].Z==1;plus&=model[l].X==2&&model[l].Y==1&&model[l].Z==1;}}
    if((!so||!p.opaque)&&!plus)return 8;p.smooth=false;p.fixed_alpha=false;
    for(int k=0;k<4*L;++k)point[k]=configured::clip(initial[k]);T sum=0;for(int l=0;l<L;++l){if(plus&&point[l]<=0)point[l]=T(1)/T(L);sum+=point[l];}
    if(plus){for(int l=0;l<L;++l)point[l]/=sum;if(p.opaque){point[0]=1;for(int l=1;l<L;++l)point[l]=0;}}else point[0]=1;
    auto cv=[](View<T> v){return View<const T>{v.data,v.stride};};View<const T> alpha{point.data,point.stride};int a=0,v=0;
    int status=improve<T,Cube>(model,p,cv(point),target,alpha,bg,point,work,tolerance,0,a,v);values=v;accepted=0;if(status!=0&&status!=5)return status;
    using P=general::Program<T,Prior,Model>;T best=P::energy(model,p,cv(point));++values;
    View<T> candidate{work.data+(Cube?size_t(26)*L:range_union::words(L))*work.stride,work.stride};
    View<T> preferred{work.data+size_t(18*L)*work.stride,work.stride},trial{work.data+size_t(22*L)*work.stride,work.stride};
    for(int outer=0;outer<rounds;++outer){
        bool color_moved=false;
        int code=improve<T,Cube>(model,p,cv(point),target,alpha,bg,candidate,work,tolerance,3,a,v);values+=v;T cost=P::energy(model,p,cv(candidate));++values;
        if((code==0||code==5)&&std::isfinite(cost)&&cost<best){for(int k=0;k<4*L;++k)point[k]=candidate[k];best=cost;accepted+=a;status=code;color_moved=true;}
        T dual[3];if(!proposal<T,Cube>(model,p,target,point,work,dual))break;
        View<T> prefix{work.data+size_t(10*L)*work.stride,work.stride},gradient{work.data+size_t(14*L)*work.stride,work.stride};
        alpha_adjoint(model,p,cv(point),cv(prefix),dual,gradient,so&&p.opaque);++values;
        T scale=1;for(int l=0;l<L;++l)if(!(p.opaque&&l==0)){T g=std::abs(gradient[l]);if(g>scale)scale=g;}if(!std::isfinite(scale))break;
        bool moved=false;T amount=T(.5);
        for(int attempt=0;attempt<6;++attempt){for(int k=0;k<4*L;++k)preferred[k]=point[k];sum=0;
            for(int l=0;l<L;++l){T next=plus?point[l]*std::exp(-amount*gradient[l]/scale):configured::clip(point[l]-amount*gradient[l]/scale);preferred[l]=next;sum+=next;}
            if(plus){for(int l=0;l<L;++l)preferred[l]/=sum;if(p.opaque){preferred[0]=1;for(int l=1;l<L;++l)preferred[l]=0;}}else preferred[0]=1;
            code=improve<T,Cube>(model,p,cv(preferred),target,cv(preferred),bg,trial,work,tolerance,0,a,v);values+=v;cost=P::energy(model,p,cv(trial));++values;
            if((code==0||code==5)&&std::isfinite(cost)&&cost<best){for(int k=0;k<4*L;++k)point[k]=trial[k];best=cost;status=code;++accepted;moved=true;break;}amount*=T(.5);
        }
        if(!moved&&!color_moved)break;
    }
    return status;
}

template<class T>__global__ void kernel(const Model* model,general::Problem p,int n,int slots,const T* initial,const T* target,const T* alpha,const T* bg,T* output,T* state,int* statuses,int* accepted,int* values,T tolerance,int rounds,int* paths=nullptr,bool inverse=true,int state_tile=0){
    int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=slots)return;View<T> work=state_view(state,slot,slots,words(p.layers,inverse),state_tile);
    for(int i=slot;i<n;i+=slots){auto cv=[&](const T* x){return View<const T>{x+i,size_t(n)};};View<T> point{output+i,size_t(n)};int code=8;
        if(inverse){if(p.refinement)code=improve(model,p,cv(initial),cv(target),cv(alpha),cv(bg),point,work,tolerance,rounds,accepted[i],values[i]);else code=first(model,p,cv(initial),cv(target),cv(bg),point,work,tolerance,rounds,accepted[i],values[i]);}
        bool coupled=code==8;if(coupled)code=coupled_sqp::Solver<T>::run(model,p,cv(initial),cv(target),cv(alpha),cv(bg),point,work,tolerance,rounds,accepted[i],values[i]);
        statuses[i]=code;if(paths)paths[i]=coupled?1:0;
    }
}

// Compile disjoint solver bodies so ordinary pixels do not inherit the other
// solver's register/call-frame requirements. Unsupported inverse jobs go into
// a device queue and are solved in the SAME float precision before validation.
// Controlled resource experiment: four 128-thread blocks per SM. This changes
// compiler allocation, not solver state/equations. More spills may offset the
// extra residency, so actual original-image quality and timings decide utility.
template<class T,int Path>__global__ __launch_bounds__(128,4) void split_kernel(const Model* model,general::Problem p,int n,int slots,const T* initial,const T* target,const T* alpha,const T* bg,T* output,T* state,int* statuses,int* accepted,int* values,T tolerance,int rounds,int* paths,bool inverse,int state_tile,int* list,int* count,bool queued){
    static_assert(Path==1||Path==2,"Invalid split solver path");
    int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=slots)return;
    View<T> work=state_view(state,slot,slots,words(p.layers,inverse),state_tile);
    int jobs=queued?*count:n;
    for(int job=slot;job<jobs;job+=slots){
        int i=queued?list[job]:job;auto cv=[&](const T* x){return View<const T>{x+i,size_t(n)};};View<T> point{output+i,size_t(n)};int code;
        if constexpr(Path==1){
            if(p.refinement)code=improve(model,p,cv(initial),cv(target),cv(alpha),cv(bg),point,work,tolerance,rounds,accepted[i],values[i]);
            else code=first(model,p,cv(initial),cv(target),cv(bg),point,work,tolerance,rounds,accepted[i],values[i]);
            if(paths)paths[i]=0;
            if(code==8)list[atomicAdd(count,1)]=i;
        }else{
            code=coupled_sqp::Solver<T>::run(model,p,cv(initial),cv(target),cv(alpha),cv(bg),point,work,tolerance,rounds,accepted[i],values[i]);
            if(paths)paths[i]=1;
        }
        statuses[i]=code;
    }
}
}
