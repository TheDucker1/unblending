#pragma once
#include <cstdint>
namespace configured {
struct Layer { float mean[3], inverse[9]; int mode,X,Y,Z; };
struct Options { int rounds=2048, period=128; float step=.01f,rho=1.f,growth=2.f; };
struct State { float residual,energy; int sm=-1; unsigned long long begin=0,end=0; };
void original_solve(int layers,const Layer*,bool opaque,const double* target,double* out,bool corrected);
double original_evaluate(int layers,const Layer*,bool opaque,const double* target,
    const double* x,const double* lambda,double rho,double* gradient,double* residual);
}
