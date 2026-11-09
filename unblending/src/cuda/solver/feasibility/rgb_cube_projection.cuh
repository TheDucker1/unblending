#pragma once
#include "reachable_interval_unions.cuh"
namespace reachable_cube {
template<class Model>HD bool supported(const Model* model,general::Problem p){if(!p.opaque||p.refinement||p.smooth)return false;for(int l=0;l<p.layers;++l){if(model[l].gray)return false;if(l&&(model[l].X!=1||model[l].Y!=1||model[l].Z!=1||model[l].mode<0||model[l].mode>11||model[l].mode==6||model[l].mode==7))return false;}return true;}
HD inline size_t words(int layers){return size_t(36)*layers;}
template<class T,class Model>HD T over(const Model& m,T a,T s,T d){
 // Keep the old expression's order, substituting the proved opaque-SO values.
 T pre=range_union::native_blend(s,d,m.mode)*a*T(1)+T(1)*a*(T(1)-T(1))*s+T(1)*T(1)*(T(1)-a)*d;return pre/T(1);
}
template<class T,class Input=T,class Output=T,class Model=general::Layer>HD int pixel(const Model* model,general::Problem p,general::View<const Input> initial,general::View<const Input> target,general::View<const Input> alpha,general::View<const Input> background,general::View<Output> output,general::View<T> work,T tolerance,T anchor=T(0),bool reuse=false,T limit=T(INFINITY)){
 int L=p.layers;auto warm=[&](int l,int r){T x=T(initial[L+3*l+r]);if(anchor!=0)x+=anchor*(T(model[l].mean[r])-x);return configured::clip(x);};
 for(int l=0;l<L;++l){Input a=alpha[l];if(!configured::isfinite(a)||a<0||a>1||(p.opaque&&l==0&&a!=1))return 4;output[l]=a;for(int r=0;r<3;++r)output[L+3*l+r]=warm(l,r);if(model[l].gray)return 8;}
 T current[3];for(int r=0;r<3;++r){Input bg=background[r];if(!configured::isfinite(bg)||bg<0||bg>1)return 4;current[r]=warm(0,r);work[6*L+r]=current[r];}
 for(int l=1;l<L;++l)for(int r=0;r<3;++r){current[r]=over(model[l],T(alpha[l]),warm(l,r),current[r]);work[6*L+3*l+r]=current[r];}
 T goal[3],desired[3];bool projected=false;for(int r=0;r<3;++r){T wanted=T(target[r]);if(!configured::isfinite(wanted))return 4;goal[r]=configured::clip(wanted);desired[r]=goal[r];projected|=goal[r]!=wanted;}
 unsigned active=7;for(int l=L-1;l>0&&active;--l){T u=T(alpha[l]),v=0,w=1-u;
  for(int r=0;r<3;++r){if(!(active&(1u<<r)))continue;T destination=work[6*L+3*(l-1)+r],color;
   T preferred=configured::isfinite(destination)?configured::clip(destination):T(.5);bool conditioned=preferred!=destination;
   if(!conditioned){auto inverse=range_union::sources(range_refinement::Nearest<T,2>{warm(l,r)},model[l].mode,destination,desired[r],u,v,w);if(inverse.nearest(warm(l,r),color)){output[L+3*l+r]=configured::clip(color);active&=~(1u<<r);continue;}}
   range_union::InteriorNearest<T> nearest(preferred,model[l].mode,l);range_union::destinations(nearest,model[l].mode,T(0),T(1),desired[r],u,v,w);
   if(!nearest.nearest(destination,destination))return 2;auto inverse=range_union::sources(range_refinement::Nearest<T,2>{warm(l,r)},model[l].mode,destination,desired[r],u,v,w);if(!inverse.nearest(warm(l,r),color))return 2;output[L+3*l+r]=configured::clip(color);desired[r]=destination;
  }
 }
 for(int r=0;r<3;++r)if(active&(1u<<r))output[L+r]=configured::clip(desired[r]);
 for(int r=0;r<3;++r)current[r]=T(output[L+r]);for(int l=1;l<L;++l)for(int r=0;r<3;++r)current[r]=over(model[l],T(alpha[l]),T(output[L+3*l+r]),current[r]);
 T error=0;for(int r=0;r<3;++r){T d=current[r]-goal[r];error+=d*d;}if(!configured::isfinite(error)||error>tolerance*tolerance)return 3;return projected?5:0;
}
}
