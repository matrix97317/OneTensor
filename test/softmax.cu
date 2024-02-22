#include <one_tensor.h>
#include <cassert>

__global__ void softmax_kernel(float *input, float *output, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        float sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            sum += expf(input[j]);
        }
        output[tid] = expf(input[tid]) / sum;
    }
}

int main()
{
    Shape problem_size(4, 4);

    OneTensor<float> cuda_input(problem_size);
    OneTensor<float> cuda_output(problem_size);

    std::vector<float> input(problem_size.nums());

    std::fill_n(input.begin(), problem_size.nums(), 0.1);

    for (int i = 0; i < cuda_input.shape.d0; i++)
    {
        for (int j = 0; j < cuda_input.shape.d1; j++)
        {
            int index = i * cuda_input.shape.d1 + j;
            cuda_input.SetHostData(input[i], index);
        }
    }

    cuda_output.FillHostData(0.0f);

    cuda_input.sync_device();
    cuda_output.sync_device();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // softmax core
    int blockSize = 128;
    int gridSize = (int)ceil(float(problem_size.nums()) / blockSize);

    softmax_kernel<<<gridSize, blockSize, 0, stream>>>(cuda_input.deviceData<float>(), cuda_output.deviceData<float>(), problem_size.nums());

    cuda_output.sync_device(false);
    cuda_output.HostDataView();
    return 0;
}