#pragma once
#include "program.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>void read(std::ifstream& f,T* p,size_t n){f.read(reinterpret_cast<char*>(p),n*sizeof(T));if(!f)throw std::runtime_error("Truncated input");}
struct Case {
    general::Problem p;int count,C;bool custom;double rho;
    std::vector<general::Layer> model;
    std::vector<double> x,target,alpha,bg,lambda;
    explicit Case(const std::string& path){std::ifstream f(path,std::ios::binary);int h[10];read(f,h,10);if(h[0]!=0x47454e31||h[1]<1||h[2]<1||h[2]>4096)throw std::runtime_error("Bad general fixture");
        count=h[1];p.layers=h[2];p.refinement=h[3];p.opaque=h[4];p.smooth=h[5];p.fixed_alpha=h[6];p.sparsity=h[7];p.minimum=h[8];custom=h[9];read(f,&rho,1);read(f,&p.sparsity_weight,1);model.resize(p.layers);
        static_assert(sizeof(general::Layer)==128,"Serialized layer layout");read(f,model.data(),model.size());C=p.constraints(model.data());
        for(auto v:{&x,&target,&alpha,&bg,&lambda}){int channels=v==&x?4*p.layers:(v==&alpha?p.layers:(v==&lambda?C:3));v->resize(size_t(channels)*count);read(f,v->data(),v->size());}
        if(f.peek()!=EOF)throw std::runtime_error("Trailing fixture data");
    }
    int output_planes()const{return 2+4*p.layers+4*p.layers+C+8*p.layers;}
};
