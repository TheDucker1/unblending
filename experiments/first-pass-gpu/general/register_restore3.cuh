#pragma once
#include "register_geometry.cuh"
#ifdef __CUDACC__
#define REG3_COLD __host__ __device__ __noinline__
#else
#define REG3_COLD __attribute__((noinline))
#endif
namespace register_three {
template<class View>REG3_HD Point load(View x){return {x[1],x[2],x[3],{x[6],x[7],x[8]},x[9]};}
template<class View>REG3_HD void store(View x,Point v){x[0]=1;x[1]=v.a;x[2]=v.u;x[3]=x[4]=x[5]=v.bg;x[6]=v.c.r;x[7]=v.c.g;x[8]=v.c.b;x[9]=x[10]=x[11]=v.top;}
REG3_HD float gray(float a,float b,float c){a=bound(a);b=bound(b);c=bound(c);return a!=b||b!=c?bound((a+b+c)/3.f):a;}
template<class View>REG3_HD Point initial(View x){return {bound(x[1]),bound(x[2]),gray(x[3],x[4],x[5]),{bound(x[6]),bound(x[7]),bound(x[8])},gray(x[9],x[10],x[11])};}
template<class Model>REG3_HD bool eligible(const Model* m,general::Problem p){
    return !p.refinement&&p.opaque&&p.layers==3&&m[0].gray&&!m[1].gray&&m[2].gray&&m[1].mode==m[2].mode&&(m[1].mode==6||m[1].mode==7)&&m[1].X==1&&m[1].Y==1&&m[1].Z==1&&m[2].X==1&&m[2].Y==1&&m[2].Z==1;
}
template<class Model>REG3_COLD bool cold_solve(Matrix m,V rhs,V& out){
    float a[16]={m.d0,m.x01,m.x02,0,m.x01,m.d1,m.x12,0,m.x02,m.x12,m.d2,0,0,0,0,0},b[3]={rhs.r,rhs.g,rhs.b},x[4]{};
    bool ok=general::ConstraintRestore<float,Model>::solve(3,a,b,x);out={x[0],x[1],x[2]};return ok;
}
template<class Model>REG3_COLD bool neutral(const Model* model,general::Problem p,V target,Point& out){
    p.layers=3;p.refinement=false;p.opaque=true;float x[12]{},alpha[3]{},bg[3]{},t[3]={target.r,target.g,target.b};general::View<float> point{x,1};using R=general::ConstraintRestore<float,Model>;
    if(!R::neutral(model,p,t,point,0))return false;R::project(model,p,{alpha,1},bg,point);out=load(point);return true;
}
template<int Mode,class Model>REG3_HD void descend(Point& current,Evaluation cache,Point& best,float& best_value,V target,int iterations,float goal){
    float value=cache.value;
    #pragma unroll 1
    for(int iteration=0;iteration<iterations;++iteration){
        if(value<best_value){best=current;best_value=value;}if(value<=goal)break;
        Jacobian j=jacobian(current,cache,Mode,Mode);Step gradient=step(current,j,cache.residual);Matrix matrix=gram(j,gradient.mask);V dual;
        bool solved=solve(matrix,cache.residual,dual);float tr=trace(matrix);if(!solved&&tr>0&&std::isfinite(tr))solved=cold_solve<Model>(matrix,cache.residual,dual);
        Step normal;if(solved)normal=step(current,j,dual);Point previous;bool previous_valid=false,moved=false;
        #pragma unroll 1
        for(int method=0;method<2&&!moved;++method){
            if(method?!(tr>0):!solved)continue;Step direction=method?gradient:normal;float amount=method?1.f/tr:1.f;
            #pragma unroll 1
            for(int back=0;back<16;++back){
                Point candidate=trial(current,direction,amount);if(identical(candidate,current))break;candidate=project(candidate);
                if(previous_valid&&identical(candidate,previous)){amount*=.5f;continue;}
                cache=evaluate(candidate,target,Mode,Mode);
                if(cache.value<value){current=candidate;value=cache.value;moved=true;break;}
                previous=candidate;previous_valid=true;amount*=.5f;
            }
        }
        if(!moved)break;
    }
    if(value<best_value){best=current;best_value=value;}
}
template<int Mode,class Model>REG3_HD Point restore(const Model* model,general::Problem p,Point start,V target,int initial_iterations,float& best_value,float tolerance=.005f){
    Point current=start,best=start;Evaluation e=evaluate(current,target,Mode,Mode);best_value=e.value;
    float goal=tolerance*tolerance*.0025f;descend<Mode,Model>(current,e,best,best_value,target,initial_iterations,goal);
    if(best_value>goal&&neutral(model,p,target,current))descend<Mode,Model>(current,evaluate(current,target,Mode,Mode),best,best_value,target,48,goal);
    return best;
}
template<class Model>REG3_HD void scalar_initial(const Model* model,general::Problem p,const float* target,general::View<float> point,float tolerance=.005f){
    Point start=load(point);V t{target[0],target[1],target[2]};float value;Point result=model[1].mode==6?restore<6>(model,p,start,t,48,value,tolerance):restore<7>(model,p,start,t,48,value,tolerance);store(point,result);
}
}
