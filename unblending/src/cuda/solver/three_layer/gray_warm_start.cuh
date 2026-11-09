#pragma once
namespace gray_warm_seed {
template<class T,class Model>HD bool run(const Model* m,general::Problem p,const T* target,general::View<const T> initial,general::View<T> out){
    if(p.layers!=3||p.smooth||initial[0]!=1||!m[0].gray||m[1].gray||!m[2].gray||m[1].mode!=m[2].mode||(m[1].mode!=6&&m[1].mode!=7))return false;
    for(int l=1;l<3;++l)if(m[l].X!=1||m[l].Y!=1||m[l].Z!=1)return false;
    bool burn=m[1].mode==7;T a=initial[1],u=initial[2],c=1-u,d=1-a;
    if(!(a>0&&a<=1&&u>0&&u<=1))return false;
    T t[3],low=1,high=0;
    for(int r=0;r<3;++r){t[r]=burn?1-target[r]:target[r];if(t[r]<0||t[r]>1)return false;if(t[r]<low)low=t[r];if(t[r]>high)high=t[r];}
    T minimum=(high-d*low)/a;if(minimum<1)minimum=1;
    T eps=range_union::epsilon<T>(),maximum=c+u/(2*eps);
    if(c>0&&high>u){
        T qhi=(high-u)/c,qlo=low>u?(low-u)/c:T(0);
        if(qhi-d*qlo>a){T denominator=qhi-a;if(!(denominator>0))return false;T upper=d*low/denominator;if(upper<maximum)maximum=upper;}
    }
    if(!(minimum<=maximum&&configured::isfinite(minimum)&&configured::isfinite(maximum)))return false;
    T warm_top=burn?1-initial[9]:initial[9];
    T warm_k=warm_top<1?c+u/(1-warm_top):maximum;
    T k=warm_k<minimum?minimum:(warm_k>maximum?maximum:warm_k);
    // A tiny inward shift avoids constructing an empty interval by roundoff
    // when the optimum projects exactly onto a feasibility boundary.
    T slack=T(16)*std::numeric_limits<T>::epsilon()*(1+k);
    if(k<minimum+slack&&minimum+slack<=maximum)k=minimum+slack;
    if(k>maximum-slack&&maximum-slack>=minimum)k=maximum-slack;
    T top=1-u/(k-c),split=u+c*(1-top),q[3],qlo=1,qhi=0;
    for(int r=0;r<3;++r){q[r]=t[r]<=split?t[r]/k:(c>0?(t[r]-u)/c:T(1)-top);if(q[r]<qlo)qlo=q[r];if(q[r]>qhi)qhi=q[r];}
    T lower=d>0?(qhi-a)/d:T(0);if(lower<2*eps)lower=2*eps;
    if(lower>qlo)return false;
    T b=burn?1-initial[3]:initial[3];b=b<lower?lower:(b>qlo?qlo:b);
    for(int l=0;l<3;++l)out[l]=initial[l];
    for(int r=0;r<3;++r){T v=(q[r]-d*b)/a;if(!(v>=b-T(32)*std::numeric_limits<T>::epsilon()&&v<=1+T(32)*std::numeric_limits<T>::epsilon()))return false;
        T s=v>0?1-b/v:T(0);if(v>=1){T warm=burn?1-initial[6+r]:initial[6+r];if(warm>s)s=warm;}
        s=configured::clip(s);out[3+r]=burn?1-b:b;out[6+r]=burn?1-s:s;out[9+r]=burn?1-top:top;
    }
    return true;
}
}
