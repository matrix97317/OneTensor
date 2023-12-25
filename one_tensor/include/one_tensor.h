#ifndef __ONE_TENSOR_H__
#define __ONE_TENSOR_H__

#include <cuda_runtime.h>
#ifdef USE_DLPACK
#include <dlpack/dlpack.h>
#endif
#include <iostream>
#include <stdexcept>
#include <string>
#include "cnpy.h"
#include "cuda_gadget.h"
#include <cuda_fp16.h>
#include <cuda/std/type_traits>
struct Shape{
    size_t nDims;
    size_t d0=1,d1=1,d2=1,d3=1,d4=1;
    Shape(){};
    Shape(const Shape &s){
        this->d0=s.d0;
        this->d1=s.d1;
        this->d2=s.d2;
        this->d3=s.d3;
        this->d4=s.d4;
        this->nDims=s.nDims;
    }
    Shape(std::vector<size_t> shape){
        size_t shape_size = shape.size();
        switch (shape_size)
        {
            case 1:
                d0=shape[0];
                nDims=1;
                break;
            case 2:
                d0=shape[0];
                d1=shape[1];
                nDims=2;
                break;
            case 3:
                d0=shape[0];
                d1=shape[1];
                d2=shape[2];
                nDims=3;
                break;
            case 4:
                d0=shape[0];
                d1=shape[1];
                d2=shape[2];
                d3=shape[3];
                nDims=4;
                break;
            case 5:
                d0=shape[0];
                d1=shape[1];
                d2=shape[2];
                d3=shape[3];
                d4=shape[4];
                nDims=5;
                break;
            default:
                throw std::runtime_error("shape_size error: "+std::to_string(shape_size));
                break;
        }

    }
    Shape(size_t D0):d0(D0){nDims=1;};
    Shape(size_t D0,size_t D1):d0(D0),d1(D1){nDims=2;};
    Shape(size_t D0,size_t D1,size_t D2):d0(D0),d1(D1),d2(D2){nDims=3;};
    Shape(size_t D0,size_t D1,size_t D2, size_t D3):d0(D0),d1(D1),d2(D2),d3(D3){nDims=4;};
    Shape(size_t D0,size_t D1,size_t D2, size_t D3,size_t D4):d0(D0),d1(D1),d2(D2),d3(D3),d4(D4){nDims=5;};

    template <typename T>
    size_t size() {
        return d0 * d1 * d2 * d3 * d4* sizeof(T);
    }
    size_t nums() {
        return d0 * d1 * d2 * d3 * d4;
    }
    size_t strides(size_t index){
        switch (index)
        {
            case 0:
                return d1 * d2 * d3 * d4;
                break;
            case 1:
                return d2 * d3 * d4;
                break;
            case 2:
                return d3 * d4;
                break;
            case 3:
                return d4;
                break;
            default:
                throw std::runtime_error("strides index error: "+std::to_string(index));
                break;
        }
    }
    std::vector<size_t> to_vector(){
        std::vector<size_t> shape_vec={d0,d1,d2,d3,d4};
        return shape_vec;
    }
};

std::ostream & operator<<( std::ostream  & os,const half & val)
{
    os << __half2float(val)+0.0;
    return os;
}
std::ostream & operator<<( std::ostream  & os,const uint8_t & val)
{
    os << val+0;
    return os;
}
std::ostream & operator<<( std::ostream  & os,const int8_t & val)
{
    os << val+0;
    return os;
}
std::ostream & operator<<( std::ostream  & os,const Shape & val)
{
    os << "[ "<<val.d0<<", "<<val.d1<<", "\
            <<val.d2<<", "<<val.d3<<", "\
            <<val.d4<<", "<<" ]";
    return os;
}

uint32_t int32_bit4(const int d0, 
                const int d1,
                const int d2,
                const int d3,
                const int d4,
                const int d5,
                const int d6,
                const int d7
                ){
    uint32_t ret = ((d0&0x0000000f)<<28)+
    ((d1&0x0000000f)<<24)+
    ((d2&0x0000000f)<<20)+
    ((d3&0x0000000f)<<16)+
    ((d4&0x0000000f)<<12)+
    ((d5&0x0000000f)<<8)+
    ((d6&0x0000000f)<<4)+
    ((d7&0x0000000f));
    return ret;
}

template <typename DT>
class OneTensor{
    public:
        OneTensor(Shape s){
            shape=s;
            host_data = (DT*)malloc(s.size<DT>());
        }
        OneTensor(std::string fname){
            std::string suffix = ".npy";
            std::string substring = fname.substr(fname.length() - suffix.length());
            if (substring!=suffix){
                throw std::runtime_error("your input file'suffix is incorrect, must be '.npy' ");
            }
            cnpy::NpyArray npy_obj = cnpy::npy_load(fname);
            if (npy_obj.shape.size() > 5){
                throw std::runtime_error("OneTensor don't support 5 dims Tensor. ");
            }
            shape=Shape(npy_obj.shape);
            host_data = (DT*)malloc(shape.size<DT>());
            auto npy_data = npy_obj.as_vec<DT>();
            for(int i=0;i<npy_data.size();i++){
                this->SetHostData(npy_data[i],i);
            }
        }
        OneTensor(std::string fname,std::string varname){
            std::string suffix = ".npz";
            std::string substring = fname.substr(fname.length() - suffix.length());
            if (substring!=suffix){
                throw std::runtime_error("your input file'suffix is incorrect, must be '.npz' ");
            }
            cnpy::NpyArray npy_obj = cnpy::npz_load(fname,varname);
            if (npy_obj.shape.size() > 5){
                throw std::runtime_error("OneTensor don't support 5 dims Tensor. ");
            }
            shape=Shape(npy_obj.shape);
            host_data = (DT*)malloc(shape.size<DT>());
            auto npy_data = npy_obj.as_vec<DT>();
            for(int i=0;i<npy_data.size();i++){
                this->SetHostData(npy_data[i],i);
            }
        }
        ~OneTensor(){
            free(host_data);
            cudaFree(device_data);
        }
        void release(){
            free(host_data);
            cudaFree(device_data);
        }
        template<typename T>
        T * deviceData(){
            if (device_data==nullptr){
                throw std::runtime_error("device_data is NULL");
            }
            return reinterpret_cast<T*>(device_data);    
        }
        template<typename T>
        T * hostData(){
            if (host_data==nullptr){
                throw std::runtime_error("host_data is NULL");
            }
            return reinterpret_cast<T*>(host_data);  
        }
        template<typename T>
        void SaveNpyFile(std::string fname){
            cnpy::npy_save<T>(fname,reinterpret_cast<T*>(host_data),shape.to_vector());
        }
        void FillHostData(DT val){
            for(size_t i=0;i<shape.nums();i++){
                host_data[i]=DT(val);
            }
        }
        void sync_device(bool send=true){
            if (device_data==nullptr){
                cuda_checker(cudaMalloc(&device_data,shape.size<DT>()));
            }
            if(send){
                cuda_checker(cudaMemcpy(device_data,host_data,shape.size<DT>(),cudaMemcpyHostToDevice));
            }else{
                cuda_checker(cudaMemcpy(host_data,device_data,shape.size<DT>(),cudaMemcpyDeviceToHost));
            }
        }
        template<typename T>
        T GetHostData(size_t index){
            return static_cast<T>(host_data[index]);
        }
        void SetHostData(DT val,size_t index){
            host_data[index]=DT(val);
        }
        void HostDataReshape(int* reshape_axis){
            if((shape.nDims !=2) && (shape.nDims !=4)){
                throw std::runtime_error("Current host data view don't support nDim:"+std::to_string(shape.nDims));
            }
            if(shape.nDims == 4){
                std::vector<size_t> shape_vec = shape.to_vector();
                size_t cnt=0;
                DT* temp_host_data = (DT*)malloc(shape.size<DT>());
                for(int i=0; i< shape_vec[reshape_axis[0]]; i++){
                    for(int j=0; j<shape_vec[reshape_axis[1]]; j++){
                        for(int k=0; k<shape_vec[reshape_axis[2]]; k++){
                            for(int l=0; l<shape_vec[reshape_axis[3]]; l++){
                                int index = i*shape.strides(reshape_axis[0])+\
                                            j*shape.strides(reshape_axis[1])+\
                                            k*shape.strides(reshape_axis[2])+\
                                            l*shape.strides(reshape_axis[3]);
                                temp_host_data[cnt] = host_data[index];
                                cnt++;
                            }
                        }
                    }
                }
                shape = Shape(shape_vec[reshape_axis[0]],shape_vec[reshape_axis[1]],shape_vec[reshape_axis[2]], shape_vec[reshape_axis[3]]);
                free(host_data);
                host_data = temp_host_data;
            }
            
        }
      
        void HostDataView(){
            if((shape.nDims !=2) && (shape.nDims !=4)){
                throw std::runtime_error("Current host data view don't support nDim:"+std::to_string(shape.nDims));
            }
            if(shape.nDims == 2){
                for(int i=0; i<shape.d0; i++){
                    for(int j=0; j<shape.d1; j++){
                        int index = i*shape.d1+j;
                        std::cout<<host_data[index]<<" ";
                    }
                    std::cout<<std::endl;
                }
                std::cout<<std::endl;
            }
            if(shape.nDims == 4){
                for(int bs=0; bs<shape.d0;bs++){
                    std::cout<<"< "<<std::endl;
                    for(int h=0; h<shape.d1; h++){
                        for(int w=0; w<shape.d2; w++){
                            std::cout<<"[ ";
                            for(int c=0; c<shape.d3; c++){
                                int index = bs*shape.strides(0)+h*shape.strides(1)+w*shape.strides(2)+c;
                                std::cout<<host_data[index]<<" ";
                            }
                            std::cout<<" ], ";
                        }
                        std::cout<<std::endl;
                    }
                    std::cout<<" >"<<std::endl;
                }
            }
        }
        Shape shape;
    private:
        DT * host_data=nullptr;
        DT * device_data=nullptr;
};


#endif