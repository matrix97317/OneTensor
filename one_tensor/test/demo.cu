#include <one_tensor.h>

int main(){
    OneTensor<int8_t> a(Shape(8,16));
    OneTensor<int8_t> b(Shape(8,16));
    OneTensor<int32_t> c(Shape(8,8));
    OneTensor<int32_t> d(Shape(8,8));
    c.FillHostData(0);
    d.FillHostData(0);
    for(int i=0; i<a.shape.d0; i++){
        for(int j=0; j<a.shape.d1; j++){
            int index = i*a.shape.d1+j;
            a.SetHostData(j,index);
            b.SetHostData(j,index);
        }
    }
    a.HostDataView();

    a.sync_device();
    b.sync_device();
    c.sync_device();
    d.sync_device();

    int8_t *  ptr = a.deviceData<int8_t>();
    
    OneTensor<float> npz_file("./test_data.npz","data");
    std::cout<<npz_file.shape<<std::endl;
    npz_file.HostDataView();

    OneTensor<float> npy_file("./test_data.npy");
    std::cout<<npy_file.shape<<std::endl;
    npy_file.HostDataView();

    return 0;
}