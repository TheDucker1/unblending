#pragma once
#include "register_restore3.cuh"
#include "algebraic_retract.cuh"
namespace register_three {
REG3_HD Point advance(Point x,Point d,float amount){return project({x.a+amount*d.a,x.u+amount*d.u,x.bg+amount*d.bg,{x.c.r+amount*d.c.r,x.c.g+amount*d.c.g,x.c.b+amount*d.c.b},x.top+amount*d.top});}
REG3_HD bool retract_channel(float& color,float middle,float bg,float a,float top,float u,float target,float anchor,int lower,int upper){
    if(!(std::abs(anchor)>1e-10f))return false;float desired,value;
    if(!algebraic_retract::Inverse<float>::destination(upper,u,top,target,middle,desired))return false;
    if(!algebraic_retract::Inverse<float>::source(lower,a,bg,desired,color,value))return false;
    if(!(value>=0&&value<=1&&std::isfinite(value)))return false;color=value;return true;
}
REG3_HD float trial_restore(Point& candidate,V target,V anchor,int lower,int upper,float tolerance=.005f){
    Evaluation e=evaluate(candidate,target,lower,upper);float norm=e.value;
    if(!(norm<=tolerance*tolerance*.0025f)){
        if(lower==0&&upper==0)return INFINITY; // Same inverse-domain guard.
        if(!retract_channel(candidate.c.r,e.middle.r,candidate.bg,candidate.a,candidate.top,candidate.u,target.r,anchor.r,lower,upper))return INFINITY;
        if(!retract_channel(candidate.c.g,e.middle.g,candidate.bg,candidate.a,candidate.top,candidate.u,target.g,anchor.g,lower,upper))return INFINITY;
        if(!retract_channel(candidate.c.b,e.middle.b,candidate.bg,candidate.a,candidate.top,candidate.u,target.b,anchor.b,lower,upper))return INFINITY;
        norm=evaluate(candidate,target,lower,upper).value;
    }
    return norm;
}
template<class Model>REG3_HD float energy(const Model* model,general::Problem p,Point x){
    float bg[3]={x.bg,x.bg,x.bg},middle[3]={x.c.r,x.c.g,x.c.b},top[3]={x.top,x.top,x.top};
    float cost=0,sum=0,square=0;cost+=1.f*general::PolynomialPrior::evaluate(model[0],bg,static_cast<float*>(nullptr));sum+=1.f;square+=1.f*1.f;
    cost+=x.a*general::PolynomialPrior::evaluate(model[1],middle,static_cast<float*>(nullptr));sum+=x.a;square+=x.a*x.a;
    cost+=x.u*general::PolynomialPrior::evaluate(model[2],top,static_cast<float*>(nullptr));sum+=x.u;square+=x.u*x.u;
    if(p.minimum)cost+=.01f*sum;if(p.sparsity)cost+=float(p.sparsity_weight)*(sum/square-1.f);return cost;
}
}
