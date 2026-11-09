#pragma once
#ifdef __CUDACC__
#define REG3_HD __host__ __device__ __forceinline__
#else
#define REG3_HD inline
#endif
namespace register_three {
struct V {float r=0,g=0,b=0;};
struct Point {float a=0,u=0,bg=0;V c;float top=0;};
struct Jacobian {V a,u,bg,c,top;};
struct Evaluation {V middle,fmiddle,ftop,residual;float value=INFINITY;};
struct Matrix {float d0=0,x01=0,x02=0,d1=0,x12=0,d2=0;};
struct Step {Point d;unsigned mask=0;};
enum Bits {A=1,U=2,BG=4,R=8,G=16,B=32,TOP=64};
REG3_HD bool identical(float a,float b){return a==b&&(a!=0||std::signbit(a)==std::signbit(b));}
REG3_HD bool identical(Point a,Point b){return identical(a.a,b.a)&&identical(a.u,b.u)&&identical(a.bg,b.bg)&&identical(a.c.r,b.c.r)&&identical(a.c.g,b.c.g)&&identical(a.c.b,b.c.b)&&identical(a.top,b.top);}
REG3_HD float native(float s,float d,int mode){
    constexpr float eps=firstpass32::blend_function_internal_epsilon;
    if(mode==6){if(d<eps)return 0;float den=1-s;if(den<eps)return 1;if(d>=den)return 1;}
    else if(mode==7){float num=1-d;if(num<eps)return 1;if(s<eps)return 0;if(num>=s)return 0;}
    return general::blend(s,d,mode);
}
REG3_HD float over(float s,float d,float a,int mode,bool native_value){
    float f=native_value?native(s,d,mode):general::blend(s,d,mode);
    return f*a*1.f+1.f*a*(1.f-1.f)*s+1.f*1.f*(1.f-a)*d;
}
REG3_HD V forward(Point x,int lower,int upper,bool native_value=true){
    return {over(x.top,over(x.c.r,x.bg,x.a,lower,native_value),x.u,upper,native_value),over(x.top,over(x.c.g,x.bg,x.a,lower,native_value),x.u,upper,native_value),over(x.top,over(x.c.b,x.bg,x.a,lower,native_value),x.u,upper,native_value)};
}
REG3_HD float over_value(float f,float s,float d,float a){return f*a*1.f+1.f*a*(1.f-1.f)*s+1.f*1.f*(1.f-a)*d;}
REG3_HD Evaluation evaluate(Point x,V target,int lower,int upper){
    Evaluation e;e.fmiddle={native(x.c.r,x.bg,lower),native(x.c.g,x.bg,lower),native(x.c.b,x.bg,lower)};
    e.middle={over_value(e.fmiddle.r,x.c.r,x.bg,x.a),over_value(e.fmiddle.g,x.c.g,x.bg,x.a),over_value(e.fmiddle.b,x.c.b,x.bg,x.a)};
    e.ftop={native(x.top,e.middle.r,upper),native(x.top,e.middle.g,upper),native(x.top,e.middle.b,upper)};
    e.residual={over_value(e.ftop.r,x.top,e.middle.r,x.u)-target.r,over_value(e.ftop.g,x.top,e.middle.g,x.u)-target.g,over_value(e.ftop.b,x.top,e.middle.b,x.u)-target.b};
    float result=0,d=e.residual.r;result+=d*d;d=e.residual.g;result+=d*d;d=e.residual.b;result+=d*d;result+=0.f;
    float length=0;length+=x.bg*x.bg;length+=x.bg*x.bg;length+=x.bg*x.bg;length=::sqrtf(length);d=float(1.7320508075688772935)*x.bg-length;result+=d*d;result+=d*d;result+=d*d;
    length=0;length+=x.top*x.top;length+=x.top*x.top;length+=x.top*x.top;length=::sqrtf(length);d=float(1.7320508075688772935)*x.top-length;result+=d*d;result+=d*d;result+=d*d;
    e.value=std::isfinite(result)?result:INFINITY;return e;
}
REG3_HD float norm(Point x,V target,int lower,int upper){return evaluate(x,target,lower,upper).value;}
REG3_HD void channel(float source,float bg,float a,float top,float u,float middle,float fmiddle,float ftop,int lower,int upper,float& ja,float& ju,float& jb,float& jc,float& jt){
    float q=u*general::grad_d(top,middle,upper)+(1-u);
    ja=q*(fmiddle-bg);ju=ftop-middle;
    jb=q*(a*general::grad_d(source,bg,lower)+(1-a));jc=q*(a*general::grad_s(source,bg,lower));jt=u*general::grad_s(top,middle,upper);
}
REG3_HD Jacobian jacobian(Point x,Evaluation e,int lower,int upper){
    Jacobian j;channel(x.c.r,x.bg,x.a,x.top,x.u,e.middle.r,e.fmiddle.r,e.ftop.r,lower,upper,j.a.r,j.u.r,j.bg.r,j.c.r,j.top.r);channel(x.c.g,x.bg,x.a,x.top,x.u,e.middle.g,e.fmiddle.g,e.ftop.g,lower,upper,j.a.g,j.u.g,j.bg.g,j.c.g,j.top.g);channel(x.c.b,x.bg,x.a,x.top,x.u,e.middle.b,e.fmiddle.b,e.ftop.b,lower,upper,j.a.b,j.u.b,j.bg.b,j.c.b,j.top.b);return j;
}
REG3_HD Jacobian jacobian(Point x,int lower,int upper){return jacobian(x,evaluate(x,{},lower,upper),lower,upper);}
REG3_HD float negative_dot(V j,V v){float x=0;x-=j.r*v.r;x-=j.g*v.g;x-=j.b*v.b;return x;}
REG3_HD float negative_gray(V j,V v){float x=0;x-=j.r*v.r/3.f;x-=j.g*v.g/3.f;x-=j.b*v.b/3.f;return x;}
REG3_HD bool free(float x,float d){return !((x<=0&&d<0)||(x>=1&&d>0));}
REG3_HD Step step(Point x,Jacobian j,V v){
    Step s;s.d={negative_dot(j.a,v),negative_dot(j.u,v),negative_gray(j.bg,v),{-j.c.r*v.r,-j.c.g*v.g,-j.c.b*v.b},negative_gray(j.top,v)};
    if(free(x.a,s.d.a))s.mask|=A;if(free(x.u,s.d.u))s.mask|=U;if(free(x.bg,s.d.bg))s.mask|=BG;if(free(x.c.r,s.d.c.r))s.mask|=R;if(free(x.c.g,s.d.c.g))s.mask|=G;if(free(x.c.b,s.d.c.b))s.mask|=B;if(free(x.top,s.d.top))s.mask|=TOP;return s;
}
template<bool Gray>REG3_HD void outer(Matrix& m,V j){
    if constexpr(Gray){m.d0+=j.r*j.r/3.f;m.x01+=j.r*j.g/3.f;m.x02+=j.r*j.b/3.f;m.d1+=j.g*j.g/3.f;m.x12+=j.g*j.b/3.f;m.d2+=j.b*j.b/3.f;}
    else{m.d0+=j.r*j.r;m.x01+=j.r*j.g;m.x02+=j.r*j.b;m.d1+=j.g*j.g;m.x12+=j.g*j.b;m.d2+=j.b*j.b;}
}
REG3_HD Matrix gram(Jacobian j,unsigned mask){
    Matrix m;if(mask&BG)outer<true>(m,j.bg);if(mask&A)outer<false>(m,j.a);
    if(mask&R)m.d0+=j.c.r*j.c.r;if(mask&G)m.d1+=j.c.g*j.c.g;if(mask&B)m.d2+=j.c.b*j.c.b;
    if(mask&U)outer<false>(m,j.u);if(mask&TOP)outer<true>(m,j.top);return m;
}
REG3_HD float trace(Matrix m){float t=0;t+=m.d0;t+=m.d1;t+=m.d2;return t;}
REG3_HD bool solve(Matrix m,V rhs,V& x){
    float t=m.d0+m.d1+m.d2;if(!(t>0&&std::isfinite(t)))return false;float scale=1/t,ridge=1e-6f;
    float a00=m.d0*scale+ridge,a11=m.d1*scale+ridge,a22=m.d2*scale+ridge,a01=(m.x01+m.x01)*.5f*scale,a02=(m.x02+m.x02)*.5f*scale,a12=(m.x12+m.x12)*.5f*scale;
    if(!(a00>ridge*.25f))return false;float l10=a01/a00,l20=a02/a00,d1=::fmaf(-l10,a01,a11),off=::fmaf(-l20,a01,a12);if(!(d1>ridge*.25f))return false;
    float l21=off/d1,d2=::fmaf(-l21,off,::fmaf(-l20,a02,a22));if(!(d2>ridge*.25f))return false;
    float y0=rhs.r*scale,y1=::fmaf(-l10,y0,rhs.g*scale),y2=::fmaf(-l21,y1,::fmaf(-l20,y0,rhs.b*scale));
    x.b=y2/d2;x.g=::fmaf(-l21,x.b,y1/d1);x.r=::fmaf(-l20,x.b,::fmaf(-l10,x.g,y0/a00));return std::isfinite(x.r)&&std::isfinite(x.g)&&std::isfinite(x.b);
}
REG3_HD float raw(float x,float d,float amount,bool active){return active?configured::clip(x+amount*d):x;}
REG3_HD Point trial(Point x,Step s,float amount){return {raw(x.a,s.d.a,amount,s.mask&A),raw(x.u,s.d.u,amount,s.mask&U),raw(x.bg,s.d.bg,amount,s.mask&BG),{raw(x.c.r,s.d.c.r,amount,s.mask&R),raw(x.c.g,s.d.c.g,amount,s.mask&G),raw(x.c.b,s.d.c.b,amount,s.mask&B)},raw(x.top,s.d.top,amount,s.mask&TOP)};}
REG3_HD float bound(float v){return std::isfinite(v)?configured::clip(v):.5f;}
REG3_HD Point project(Point x){return {bound(x.a),bound(x.u),bound(x.bg),{bound(x.c.r),bound(x.c.g),bound(x.c.b)},bound(x.top)};}
}
