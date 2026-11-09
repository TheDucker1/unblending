#pragma once
#include "trial_step.cuh"
namespace register_three {
struct FullMatrix {V r,g,b;};
struct PriorMetric {V gb,gc,gt;float ga,gu,bb,tt,ab,ub;Matrix color;};
REG3_HD bool finite(V v){return std::isfinite(v.r)&&std::isfinite(v.g)&&std::isfinite(v.b);}
REG3_HD V plus(V g,V j,V v){return {g.r+j.r*v.r,g.g+j.g*v.g,g.b+j.b*v.b};}
REG3_HD float weighted(V g,float b,float f){float v=0;v+=b*f*f*g.r;v+=b*f*f*g.g;v+=b*f*f*g.b;return v;}
REG3_HD float negative_weighted(V g,float b,float f){float v=0;v-=b*f*f*g.r;v-=b*f*f*g.g;v-=b*f*f*g.b;return v;}
REG3_HD void color_gram(FullMatrix& g,V j,Matrix b,V f){
 g.r.r+=j.r*b.d0*f.r*f.r*j.r;g.r.g+=j.r*b.x01*f.r*f.g*j.g;g.r.b+=j.r*b.x02*f.r*f.b*j.b;
 g.g.r+=j.g*b.x01*f.g*f.r*j.r;g.g.g+=j.g*b.d1*f.g*f.g*j.g;g.g.b+=j.g*b.x12*f.g*f.b*j.b;
 g.b.r+=j.b*b.x02*f.b*f.r*j.r;g.b.g+=j.b*b.x12*f.b*f.g*j.g;g.b.b+=j.b*b.d2*f.b*f.b*j.b;
}
REG3_HD void alpha_gram(FullMatrix& g,V j,float b){
 g.r.r+=j.r*b*j.r;g.r.g+=j.r*b*j.g;g.r.b+=j.r*b*j.b;
 g.g.r+=j.g*b*j.r;g.g.g+=j.g*b*j.g;g.g.b+=j.g*b*j.b;
 g.b.r+=j.b*b*j.r;g.b.g+=j.b*b*j.g;g.b.b+=j.b*b*j.b;
}
REG3_HD V color_product(Matrix b,V f,V g){V v;
 v.r+=b.d0*f.r*f.r*g.r;v.r+=b.x01*f.r*f.g*g.g;v.r+=b.x02*f.r*f.b*g.b;
 v.g+=b.x01*f.g*f.r*g.r;v.g+=b.d1*f.g*f.g*g.g;v.g+=b.x12*f.g*f.b*g.b;
 v.b+=b.x02*f.b*f.r*g.r;v.b+=b.x12*f.b*f.g*g.g;v.b+=b.d2*f.b*f.b*g.b;return v;
}
REG3_HD V negative_color_product(Matrix b,V f,V g){V v;
 v.r-=b.d0*f.r*f.r*g.r;v.r-=b.x01*f.r*f.g*g.g;v.r-=b.x02*f.r*f.b*g.b;
 v.g-=b.x01*f.g*f.r*g.r;v.g-=b.d1*f.g*f.g*g.g;v.g-=b.x12*f.g*f.b*g.b;
 v.b-=b.x02*f.b*f.r*g.r;v.b-=b.x12*f.b*f.g*g.g;v.b-=b.d2*f.b*f.b*g.b;return v;
}
REG3_HD bool solve_full(FullMatrix m,V rhs,V& x){
 float t=m.r.r+m.g.g+m.b.b;if(!(t>0&&std::isfinite(t)))return false;float scale=1/t,ridge=1e-6f;
 float a00=m.r.r*scale+ridge,a11=m.g.g*scale+ridge,a22=m.b.b*scale+ridge,a01=(m.r.g+m.g.r)*.5f*scale,a02=(m.r.b+m.b.r)*.5f*scale,a12=(m.g.b+m.b.g)*.5f*scale;
 if(!(a00>ridge*.25f))return false;float l10=a01/a00,l20=a02/a00,d1=::fmaf(-l10,a01,a11),off=::fmaf(-l20,a01,a12);if(!(d1>ridge*.25f))return false;
 float l21=off/d1,d2=::fmaf(-l21,off,::fmaf(-l20,a02,a22));if(!(d2>ridge*.25f))return false;
 float y0=rhs.r*scale,y1=::fmaf(-l10,y0,rhs.g*scale),y2=::fmaf(-l21,y1,::fmaf(-l20,y0,rhs.b*scale));
 x.b=y2/d2;x.g=::fmaf(-l21,x.b,y1/d1);x.r=::fmaf(-l20,x.b,::fmaf(-l10,x.g,y0/a00));return finite(x);
}
template<class Model>REG3_COLD bool cold_full(FullMatrix m,V rhs,V& out){float a[16]={m.r.r,m.r.g,m.r.b,0,m.g.r,m.g.g,m.g.b,0,m.b.r,m.b.g,m.b.b,0,0,0,0,0},b[3]={rhs.r,rhs.g,rhs.b},x[4]{};bool ok=general::ConstraintRestore<float,Model>::solve(3,a,b,x);out={x[0],x[1],x[2]};return ok;}
template<class Model>REG3_HD bool prior_metric(const Model* m,general::Problem p,Point x,PriorMetric& z){
 float cb[3]={x.bg,x.bg,x.bg},cc[3]={x.c.r,x.c.g,x.c.b},ct[3]={x.top,x.top,x.top},gb[3],gc[3],gt[3],cov[6],gray;
 float db=custom_refinement::local_metric(m[0],cb,gb,cov,&gray);z.gb={1.f*gb[0],1.f*gb[1],1.f*gb[2]};z.bb=gray/1.f;if(!std::isfinite(z.bb))z.bb=1.f/3.f;
 float dc=custom_refinement::local_metric(m[1],cc,gc,cov,&gray),sc=x.a>.05f?x.a:.05f;z.gc={x.a*gc[0],x.a*gc[1],x.a*gc[2]};z.color={.5f*cov[0]/sc,.5f*cov[1]/sc,.5f*cov[2]/sc,.5f*cov[3]/sc,.5f*cov[4]/sc,.5f*cov[5]/sc};
 if(!(std::isfinite(z.color.d0)&&std::isfinite(z.color.x01)&&std::isfinite(z.color.x02)&&std::isfinite(z.color.d1)&&std::isfinite(z.color.x12)&&std::isfinite(z.color.d2)))z.color={1,0,0,1,0,1};
 float dt=custom_refinement::local_metric(m[2],ct,gt,cov,&gray),st=x.u>.05f?x.u:.05f;z.gt={x.u*gt[0],x.u*gt[1],x.u*gt[2]};z.tt=gray/st;if(!std::isfinite(z.tt))z.tt=1.f/3.f;
 z.ab=1.f/(1+std::abs(dc));z.ub=1.f/(1+std::abs(dt));z.ga=dc;z.gu=dt;
 if(p.minimum){db+=.01f;z.ga+=.01f;z.gu+=.01f;}
 if(p.sparsity){float sum=0,square=0;sum+=1.f;square+=1.f;sum+=x.a;square+=x.a*x.a;sum+=x.u;square+=x.u*x.u;if(!(square>0))return false;float w=float(p.sparsity_weight);db+=w*(square-2.f*1.f*sum)/(square*square);z.ga+=w*(square-2.f*x.a*sum)/(square*square);z.gu+=w*(square-2.f*x.u*sum)/(square*square);}
 return std::isfinite(db)&&std::isfinite(z.ga)&&std::isfinite(z.gu)&&finite(z.gb)&&finite(z.gc)&&finite(z.gt);
}
REG3_HD float alpha_step(float gradient,V j,V dual,float b,float f){float v=gradient;v+=j.r*dual.r;v+=j.g*dual.g;v+=j.b*dual.b;return -b*f*v;}
template<class Model>REG3_HD bool proposal(const Model* model,general::Problem p,Point x,V target,Point& d,V& anchor,int lower,int upper){
 PriorMetric z;if(!prior_metric(model,p,x,z))return false;Evaluation e=evaluate(x,target,lower,upper);Jacobian j=jacobian(x,e,lower,upper);anchor=j.c;unsigned mask=127;
 #pragma unroll 1
 for(int pass=0;pass<4;++pass){float fb=bool(mask&BG),fa=bool(mask&A),fu=bool(mask&U),ft=bool(mask&TOP);V f{float(bool(mask&R)),float(bool(mask&G)),float(bool(mask&B))};
  FullMatrix m{};V rhs=e.residual;float b=weighted(z.gb,z.bb,fb);rhs.r-=j.bg.r*b;rhs.g-=j.bg.g*b;rhs.b-=j.bg.b*b;color_gram(m,j.bg,{z.bb,z.bb,z.bb,z.bb,z.bb,z.bb},{fb,fb,fb});
  V c=color_product(z.color,f,z.gc);rhs.r-=j.c.r*c.r;rhs.g-=j.c.g*c.g;rhs.b-=j.c.b*c.b;color_gram(m,j.c,z.color,f);
  float ab=z.ab*fa;if(ab!=0){rhs.r-=j.a.r*ab*z.ga;rhs.g-=j.a.g*ab*z.ga;rhs.b-=j.a.b*ab*z.ga;alpha_gram(m,j.a,ab);}
  b=weighted(z.gt,z.tt,ft);rhs.r-=j.top.r*b;rhs.g-=j.top.g*b;rhs.b-=j.top.b*b;color_gram(m,j.top,{z.tt,z.tt,z.tt,z.tt,z.tt,z.tt},{ft,ft,ft});
  float ub=z.ub*fu;if(ub!=0){rhs.r-=j.u.r*ub*z.gu;rhs.g-=j.u.g*ub*z.gu;rhs.b-=j.u.b*ub*z.gu;alpha_gram(m,j.u,ub);}
  float tr=0;tr+=m.r.r;tr+=m.g.g;tr+=m.b.b;V dual{};if(tr>0&&!solve_full(m,rhs,dual)&&!cold_full<Model>(m,rhs,dual))return false;if(!std::isfinite(tr)||tr<0)return false;
  d={alpha_step(z.ga,j.a,dual,z.ab,fa),alpha_step(z.gu,j.u,dual,z.ub,fu),negative_weighted(plus(z.gb,j.bg,dual),z.bb,fb),negative_color_product(z.color,f,plus(z.gc,j.c,dual)),negative_weighted(plus(z.gt,j.top,dual),z.tt,ft)};
  unsigned next=mask;if((mask&A)&&!free(x.a,d.a))next&=~A;if((mask&U)&&!free(x.u,d.u))next&=~U;if((mask&BG)&&!free(x.bg,d.bg))next&=~BG;if((mask&R)&&!free(x.c.r,d.c.r))next&=~R;if((mask&G)&&!free(x.c.g,d.c.g))next&=~G;if((mask&B)&&!free(x.c.b,d.c.b))next&=~B;if((mask&TOP)&&!free(x.top,d.top))next&=~TOP;if(next==mask)return true;mask=next;
 }
 if(!free(x.a,d.a))d.a=0;if(!free(x.u,d.u))d.u=0;if(!free(x.bg,d.bg))d.bg=0;if(!free(x.c.r,d.c.r))d.c.r=0;if(!free(x.c.g,d.c.g))d.c.g=0;if(!free(x.c.b,d.c.b))d.c.b=0;if(!free(x.top,d.top))d.top=0;return true;
}
REG3_HD bool limit_coordinate(float x,float d,float& limit,float& magnitude){if(!std::isfinite(d))return false;float a=std::abs(d);if(a>magnitude)magnitude=a;if(d>0){float t=(1-x)/d;if(t<limit)limit=t;}else if(d<0){float t=-x/d;if(t<limit)limit=t;}return true;}
REG3_HD bool limit_step(Point x,Point d,float& limit){float magnitude=0;limit=1;
 if(!(limit_coordinate(x.a,d.a,limit,magnitude)&&limit_coordinate(x.u,d.u,limit,magnitude)&&limit_coordinate(x.bg,d.bg,limit,magnitude)&&limit_coordinate(x.c.r,d.c.r,limit,magnitude)&&limit_coordinate(x.c.g,d.c.g,limit,magnitude)&&limit_coordinate(x.c.b,d.c.b,limit,magnitude)&&limit_coordinate(x.top,d.top,limit,magnitude)))return false;
 if(magnitude<=1e-10f||!(limit>0))return false;if(magnitude*limit>.25f)limit=.25f/magnitude;return true;
}
}
