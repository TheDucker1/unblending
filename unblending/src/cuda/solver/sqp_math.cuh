#pragma once
namespace sqp_math {
template<class T>HD T fused(T a,T b,T c){if constexpr(std::is_same_v<T,float>)return ::fmaf(a,b,c);else return ::fma(a,b,c);}
// A sum of J B J^T blocks is PSD. The original trace-scaled ridge makes it
// positive definite; a fixed LDL^T solve needs no data-dependent pivoting.
template<class T>HD bool spd3(const T* g,const T* rhs,T* x){
    T trace=g[0]+g[5]+g[10];if(!(trace>0&&configured::isfinite(trace)))return false;
    T scale=1/trace,ridge=sizeof(T)==4?T(1e-6):T(1e-12);
    T a00=g[0]*scale+ridge,a11=g[5]*scale+ridge,a22=g[10]*scale+ridge;
    T a01=(g[1]+g[4])*T(.5)*scale,a02=(g[2]+g[8])*T(.5)*scale,a12=(g[6]+g[9])*T(.5)*scale;
    if(!(a00>ridge*T(.25)))return false;
    T l10=a01/a00,l20=a02/a00,d1=fused(-l10,a01,a11),off=fused(-l20,a01,a12);
    if(!(d1>ridge*T(.25)))return false;T l21=off/d1,d2=fused(-l21,off,fused(-l20,a02,a22));
    if(!(d2>ridge*T(.25)))return false;
    T y0=rhs[0]*scale,y1=fused(-l10,y0,rhs[1]*scale),y2=fused(-l21,y1,fused(-l20,y0,rhs[2]*scale));
    x[2]=y2/d2;x[1]=fused(-l21,x[2],y1/d1);x[0]=fused(-l20,x[2],fused(-l10,x[1],y0/a00));
    return configured::isfinite(x[0])&&configured::isfinite(x[1])&&configured::isfinite(x[2]);
}
template<class Model>HD bool opaque_so(const Model* m,general::Problem p){
    if(!p.opaque)return false;for(int l=1;l<p.layers;++l)if(m[l].X!=1||m[l].Y!=1||m[l].Z!=1)return false;return true;
}
template<class Model>HD int rows(const Model* m,general::Problem p){return p.refinement||opaque_so(m,p)?3:4;}
template<class T,class Model>HD void jacobian(const Model* m,general::Problem p,general::View<const T> x,general::View<const T> prefix,general::View<T> J){
    bool so=opaque_so(m,p);if(!so&&!p.refinement){general::rgba_jacobian(m,p,x,prefix,J,true);return;}
    int L=p.layers,N=4*L;T q[3]={1,1,1};
    for(int l=L-1;l>0;--l){T a=x[l],d=prefix[4*(l-1)+3],A=prefix[4*l+3];J[3*N+l]=0;
        for(int r=0;r<3;++r){T s=x[L+3*l+r],c=prefix[4*(l-1)+r],ds,dc;
            if(so){ds=a*general::grad_s(s,c,m[l].mode);dc=a*general::grad_d(s,c,m[l].mode)+(1-a);}
            else{ds=a*d*general::grad_s(s,c,m[l].mode)+T(m[l].Y)*a*(1-d);dc=a*d*general::grad_d(s,c,m[l].mode)+T(m[l].Z)*d*(1-a);if(A>T(1e-12)){ds/=A;dc/=A;}}
            J[r*N+l]=p.refinement?T(0):q[r]*(general::blend(s,c,m[l].mode)-c);
            J[r*N+L+3*l+r]=q[r]*ds;q[r]*=dc;
        }}
    J[3*N]=0;for(int r=0;r<3;++r){J[r*N]=0;J[r*N+L+r]=q[r];}
}
}
