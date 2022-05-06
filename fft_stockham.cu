// nvcc -lcufft fft_stockham.cu && ./a.out

// High Performance Discrete Fourier Transforms on Graphics Processors

#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <assert.h>

#include <cufft.h>

using namespace std;

#define DEBUG 1

#define INPUT_SIZE (1 << 24)

// #define SHARED_MEMORY 1

#define NUM_THREAD 1024
__constant__ float SQRT_1_2 = 0.707106781188f; // cos(Pi/4)
__constant__ float COS_8 = 0.923879532511f;    // cos(Pi/8)
__constant__ float SIN_8 = 0.382683432365f;    // sin(Pi/8)

#define R 16
// const int R = 4;

// #define USE_CUDA_MALLOC_MANAGED 1

// static __global__ void ComplexPointwiseMulAndScale(cufftComplex *a, cufftComplex *b, long size, float scale);

void runTest();

// mul_p*q*(a) returns a*EXP(-I*Pi*P/Q)
__device__ cufftComplex mul_p1q2(cufftComplex a);
__device__ cufftComplex mul_p1q4(cufftComplex a);
__device__ cufftComplex mul_p3q4(cufftComplex a);
__device__ cufftComplex mul_p1q8(cufftComplex a);
__device__ cufftComplex mul_p3q8(cufftComplex a);
__device__ cufftComplex mul_p5q8(cufftComplex a);
__device__ cufftComplex mul_p7q8(cufftComplex a);

#define mul_p0q1(a) (a)
#define mul_p0q2 mul_p0q1
#define mul_p0q4 mul_p0q1
#define mul_p0q8 mul_p0q1
#define mul_p2q4 mul_p1q2
#define mul_p2q8 mul_p1q4
#define mul_p4q8 mul_p1q2
#define mul_p6q8 mul_p3q4
#define DFT2_TWIDDLE(a, b, t)                \
    {                                        \
        cufftComplex tmp = t(cuCsubf(a, b)); \
        a = cuCaddf(a, b);                   \
        b = tmp;                             \
    }
// from http://www.bealto.com/gpu-fft_opencl-2.html

cufftComplex *CPU_FFT(long N, cufftComplex *data0, cufftComplex *data1);
void cpu_swap(cufftComplex *&a, cufftComplex *&b);
long cpu_expand(long idxL, long N1, long N2);
void cpu_FFT(cufftComplex *v);
void cpu_FFT_2(cufftComplex *v);
void cpu_FfIteration(long j, long N, long Ns, cufftComplex *data0, cufftComplex *data1);

void GPU_FFT(long N, cufftComplex *dataI, cufftComplex *dataO);
__global__ void GPU_FFT_kernel(long N, long Ns, cufftComplex *dataI, cufftComplex *dataO);
__device__ void FfIteration(long j, long N, long Ns, cufftComplex *data0, cufftComplex *data1);
__host__ void gpu_swap(cufftComplex *&a, cufftComplex *&b);
__device__ long expand(long idxL, long N1, long N2);

void GPU_FftShMem(int N, cufftComplex *dataI, cufftComplex *dataO);
__global__ void GPU_FftShMem_kernel(int N, cufftComplex *data);
__device__ void DoFft_ShMem(cufftComplex *v, int N, int j, float *shared_r, float *shared_i);
__device__ void exchange(cufftComplex *v, int idxD, int incD, int idxS, int incS, float *shared_r, float *shared_i);

__device__ void FFT(cufftComplex *v);
__device__ void FFT_2(cufftComplex *v);
__device__ void FFT_4(cufftComplex *v);
__device__ void FFT_8(cufftComplex *v);
__device__ void FFT_16(cufftComplex *v);

long double l2err(long N, const cufftComplex *v1, const cufftComplex *v2);

inline cudaError_t checkCuda(cudaError_t result, int k)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
    {
        printf("On line: %d\n", k);

        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

long double l2err(long N, const cufftComplex *v1, const cufftComplex *v2)
{
    long double sum1 = 0, sum2 = 0;
    for (size_t i = 0; i < N; ++i)
    {
        long double dr = v1[i].x - v2[i].x,
                    di = v1[i].y - v2[i].y;
        long double t1 = sqrt(dr * dr + di * di), t2 = cuCabsf(v1[i]);
        sum1 += t1 * t1;
        sum2 += t2 * t2;
    }
    return sqrt(sum1 / sum2);
}

__device__ cufftComplex mul_p1q2(cufftComplex a)
{
    // Return a*EXP(-I*Pi*1/2) = a*(-I)
    return make_cuFloatComplex(a.y, -a.x);
}

__device__ cufftComplex mul_p1q4(cufftComplex a)
{
    // Return a*EXP(-I*Pi*1/4)
    return make_cuFloatComplex(SQRT_1_2 * (a.x + a.y), SQRT_1_2 * (-a.x + a.y));
}

__device__ cufftComplex mul_p3q4(cufftComplex a)
{
    // Return a*EXP(-I*Pi*3/4)
    return make_cuFloatComplex(SQRT_1_2 * (-a.x + a.y), SQRT_1_2 * (-a.x - a.y));
}

__device__ cufftComplex mul_p1q8(cufftComplex a)
{
    // Return a*EXP(-I*Pi*1/8)
    return cuCmulf(make_cuFloatComplex(COS_8, -SIN_8), a);
}

__device__ cufftComplex mul_p3q8(cufftComplex a)
{
    // Return a*EXP(-I*Pi*3/8)
    return cuCmulf(make_cuFloatComplex(SIN_8, -COS_8), a);
}

__device__ cufftComplex mul_p5q8(cufftComplex a)
{
    // Return a*EXP(-I*Pi*5/8)
    return cuCmulf(make_cuFloatComplex(-SIN_8, -COS_8), a);
}

__device__ cufftComplex mul_p7q8(cufftComplex a)
{
    // Return a*EXP(-I*Pi*7/8)
    return cuCmulf(make_cuFloatComplex(-COS_8, -SIN_8), a);
}

// static __global__ void ComplexPointwiseMulAndScale(cufftComplex *a, cufftComplex *b, long size, float scale)
// {
//     const long numThreads = blockDim.x * gridDim.x;
//     const long threadID = blockIdx.x * blockDim.x + threadIdx.x;
//     cufftComplex c;
//     for (long i = threadID; i < size; i += numThreads)
//     {
//         c = cuCmulf(a[i], b[i]);
//         b[i] = make_cuFloatComplex(scale * cuCrealf(c), scale * cuCimagf(c));
//     }
//     return;
// }

int get_T(int x)
{
    int i = 1;
    int a = R;
    while (a < x)
    {
        a *= R;
        i++;
    }
    return i - 1;
}

void GPU_FftShMem(int N, cufftComplex *dataI, cufftComplex *dataO)
{
    int T = (N / R > get_T(64)) ? N / R : get_T(64); // max
    int BlockX = N / T;

    dim3 dimgrid(BlockX, 1, 1);
    dim3 dimblock(T, 1, 1);

    GPU_FftShMem_kernel<<<dimgrid, dimblock>>>(N, dataI);
    dataO = dataI;
    return;
}

__global__ void GPU_FftShMem_kernel(int N, cufftComplex *data)
{
    __shared__ float shared_r[INPUT_SIZE];
    __shared__ float shared_i[INPUT_SIZE];
    cufftComplex v[R];
    int idxG = blockIdx.x * blockDim.x + threadIdx.x; // b * T + t

    if (idxG < N / R)
    {
        for (int r = 0; r < R; r++)
        {
            v[r] = data[idxG + r * blockDim.x];
        }

        if (blockDim.x == N / R)
        {
            DoFft_ShMem(v, N, threadIdx.x, shared_r, shared_i);
        }
        else
        {
            int idx = expand(threadIdx.x, N / R, R);
            exchange(v, idx, N / R, threadIdx.x, blockDim.x, shared_r, shared_i);
            DoFft_ShMem(v, N, threadIdx.x, shared_r, shared_i);
            exchange(v, threadIdx.x, blockDim.x, idx, N / R, shared_r, shared_i);
        }
        for (int r = 0; r < R; r++)
        {
            data[idxG + r * blockDim.x] = v[r];
        }
    }
}

__device__ void DoFft_ShMem(cufftComplex *v, int N, int j, float *shared_r, float *shared_i)
{
    for (int Ns = 0; Ns < N; Ns *= R)
    {
        float angle = -2 * M_PI * (j % Ns) / (Ns * R);
        for (int r = 0; r < R; r++)
        {
            v[r] = cuCmulf(v[r], make_cuFloatComplex(cos(r * angle), sin(r * angle)));
        }
        FFT(v);
        int idxD = expand(j, Ns, R);
        int idxS = expand(j, N / R, R);
        exchange(v, idxD, Ns, idxS, N / R, shared_r, shared_i);
    }
}

__device__ void exchange(cufftComplex *v, int idxD, int incD, int idxS, int incS, float *shared_r, float *shared_i)
{
#ifdef SHARED_MEMORY
    float *sr = shared_r;
    float *si = shared_i;
    __syncthreads();
    for (int r = 0; r < R; r++)
    {
        int i = idxD + r * incD;
        sr[i] = v[r].x;
        si[i] = v[r].y;
    }
    __syncthreads();
    for (int r = 0; r < R; r++)
    {
        int i = incS + r * incS;
        v[r].x = sr[i];
        v[r].y = si[i];
    }

#else
    printf("Something is wrong here. SHARED_MEMORY is not defined.\n");
#endif
}

void GPU_FFT(long N, cufftComplex *dataI, cufftComplex *dataO)
{
    long T = (N / R < NUM_THREAD) ? N / R : NUM_THREAD;                             // min
    int BlockX = ((N - 1) / (R * T) + 1 < 65536) ? ((N - 1) / (R * T) + 1) : 65536; // min

    dim3 dimgrid(BlockX, 1, 1);
    dim3 dimblock(T, 1, 1);

    for (long Ns = 1; Ns < N; Ns *= R)
    {
        GPU_FFT_kernel<<<dimgrid, dimblock>>>(N, Ns, dataI, dataO);
        cudaDeviceSynchronize();

        // global swap
        gpu_swap(dataI, dataO);
    }

    checkCuda(cudaMemcpyAsync(dataO, dataI, sizeof(cufftComplex) * N, cudaMemcpyDeviceToDevice), 205);

    return;
}
__global__ void GPU_FFT_kernel(long N, long Ns, cufftComplex *dataI, cufftComplex *dataO)
{
    long j = blockIdx.x * blockDim.x + threadIdx.x; // b * T + t

    if (j < N / R)
    {
        FfIteration(j, N, Ns, dataI + blockIdx.y * N, dataO + blockIdx.y * N);
    }
}

__device__ void FfIteration(long j, long N, long Ns, cufftComplex *data0, cufftComplex *data1)
{
    cufftComplex v[R];
    long idxS = j;
    float angle = -2 * M_PI * (j % Ns) / (Ns * R);
    for (int r = 0; r < R; r++)
    {
        v[r] = data0[idxS + r * N / R];
        v[r] = cuCmulf(v[r], make_cuFloatComplex(cos(r * angle), sin(r * angle)));
    }

    FFT(v);

    long idxD = expand(j, Ns, R);
    for (int r = 0; r < R; r++)
    {
        data1[idxD + r * Ns] = v[r];
    }
}

__device__ long expand(long idxL, long N1, long N2)
{
    return (idxL / N1) * N1 * N2 + (idxL % N1);
}

__device__ void FFT(cufftComplex *v)
{
    if (R == 2)
    {
        FFT_2(v);
    }

    if (R == 4)
    {
        FFT_4(v);
    }

    if (R == 8)
    {
        FFT_8(v);
    }

    if (R == 16)
    {
        FFT_16(v);
    }
}

__device__ void FFT_2(cufftComplex *v)
{
    cufftComplex v0 = v[0];
    v[0] = cuCaddf(v0, v[1]);
    v[1] = cuCsubf(v0, v[1]);
}

__device__ void FFT_4(cufftComplex *v)
{
    cufftComplex v0 = v[0];
    cufftComplex v1 = v[1];
    cufftComplex v2 = v[2];
    cufftComplex v3 = v[3];

    cufftComplex u0 = cuCaddf(v0, v2);
    cufftComplex u1 = cuCsubf(v0, v2);
    cufftComplex u2 = cuCaddf(v1, v3);
    cufftComplex u3 = cuCsubf(v1, v3);
    u3 = mul_p1q2(u3);

    v[0] = cuCaddf(u0, u2);
    v[1] = cuCaddf(u1, u3);
    v[2] = cuCsubf(u0, u2);
    v[3] = cuCsubf(u1, u3);
}

__device__ void FFT_8(cufftComplex *v)
{
    cufftComplex v0 = v[0];
    cufftComplex v1 = v[1];
    cufftComplex v2 = v[2];
    cufftComplex v3 = v[3];
    cufftComplex v4 = v[4];
    cufftComplex v5 = v[5];
    cufftComplex v6 = v[6];
    cufftComplex v7 = v[7];

    cufftComplex u0 = cuCaddf(v0, v4);
    cufftComplex u4 = cuCsubf(v0, v4); // mul_p0q4
    cufftComplex u1 = cuCaddf(v1, v5);
    cufftComplex u5 = mul_p1q4(cuCsubf(v1, v5));
    cufftComplex u2 = cuCaddf(v2, v6);
    cufftComplex u6 = mul_p1q2(cuCsubf(v2, v6)); // mul_p2q4
    cufftComplex u3 = cuCaddf(v3, v7);
    cufftComplex u7 = mul_p3q4(cuCsubf(v3, v7));

    v0 = cuCaddf(u0, u2);
    v2 = cuCsubf(u0, u2); // mul_p0q2
    v1 = cuCaddf(u1, u3);
    v3 = mul_p1q2(cuCsubf(u1, u3));
    v4 = cuCaddf(u4, u6);
    v6 = cuCsubf(u4, u6); // mul_p0q2
    v5 = cuCaddf(u5, u7);
    v7 = mul_p1q2(cuCsubf(u5, u7));

    v[0] = cuCaddf(v0, v1);
    v[1] = cuCaddf(v4, v5);
    v[2] = cuCaddf(v2, v3);
    v[3] = cuCaddf(v6, v7);
    v[4] = cuCsubf(v0, v1);
    v[5] = cuCsubf(v4, v5);
    v[6] = cuCsubf(v2, v3);
    v[7] = cuCsubf(v6, v7);
}

__device__ void FFT_16(cufftComplex *v)
{
    cufftComplex v0 = v[0];
    cufftComplex v1 = v[1];
    cufftComplex v2 = v[2];
    cufftComplex v3 = v[3];
    cufftComplex v4 = v[4];
    cufftComplex v5 = v[5];
    cufftComplex v6 = v[6];
    cufftComplex v7 = v[7];
    cufftComplex v8 = v[8];
    cufftComplex v9 = v[9];
    cufftComplex v10 = v[10];
    cufftComplex v11 = v[11];
    cufftComplex v12 = v[12];
    cufftComplex v13 = v[13];
    cufftComplex v14 = v[14];
    cufftComplex v15 = v[15];

    // 8x in-place DFT2 and twiddle (1)

    DFT2_TWIDDLE(v0, v8, mul_p0q8);
    DFT2_TWIDDLE(v1, v9, mul_p1q8);
    DFT2_TWIDDLE(v2, v10, mul_p2q8);
    DFT2_TWIDDLE(v3, v11, mul_p3q8);
    DFT2_TWIDDLE(v4, v12, mul_p4q8);
    DFT2_TWIDDLE(v5, v13, mul_p5q8);
    DFT2_TWIDDLE(v6, v14, mul_p6q8);
    DFT2_TWIDDLE(v7, v15, mul_p7q8);

    // 8x in-place DFT2 and twiddle (2)
    DFT2_TWIDDLE(v0, v4, mul_p0q4);
    DFT2_TWIDDLE(v1, v5, mul_p1q4);
    DFT2_TWIDDLE(v2, v6, mul_p2q4);
    DFT2_TWIDDLE(v3, v7, mul_p3q4);
    DFT2_TWIDDLE(v8, v12, mul_p0q4);
    DFT2_TWIDDLE(v9, v13, mul_p1q4);
    DFT2_TWIDDLE(v10, v14, mul_p2q4);
    DFT2_TWIDDLE(v11, v15, mul_p3q4);

    // 8x in-place DFT2 and twiddle (3)
    DFT2_TWIDDLE(v0, v2, mul_p0q2);
    DFT2_TWIDDLE(v1, v3, mul_p1q2);
    DFT2_TWIDDLE(v4, v6, mul_p0q2);
    DFT2_TWIDDLE(v5, v7, mul_p1q2);
    DFT2_TWIDDLE(v8, v10, mul_p0q2);
    DFT2_TWIDDLE(v9, v11, mul_p1q2);
    DFT2_TWIDDLE(v12, v14, mul_p0q2);
    DFT2_TWIDDLE(v13, v15, mul_p1q2);

    v[0] = cuCaddf(v0, v1);
    v[1] = cuCaddf(v8, v9);
    v[2] = cuCaddf(v4, v5);
    v[3] = cuCaddf(v12, v13);
    v[4] = cuCaddf(v2, v3);
    v[5] = cuCaddf(v10, v11);
    v[6] = cuCaddf(v6, v7);
    v[7] = cuCaddf(v14, v15);
    v[8] = cuCsubf(v0, v1);
    v[9] = cuCsubf(v8, v9);
    v[10] = cuCsubf(v4, v5);
    v[11] = cuCsubf(v12, v13);
    v[12] = cuCsubf(v2, v3);
    v[13] = cuCsubf(v10, v11);
    v[14] = cuCsubf(v6, v7);
    v[15] = cuCsubf(v14, v15);
}

__host__ void gpu_swap(cufftComplex *&a, cufftComplex *&b)
{
    cufftComplex *temp = a;
    a = b;
    b = temp;
}

void cpu_swap(cufftComplex *&a, cufftComplex *&b)
{
    cufftComplex *temp = a;
    a = b;
    b = temp;
}

long cpu_expand(long idxL, long N1, long N2)
{
    return (idxL / N1) * N1 * N2 + (idxL % N1);
}

void cpu_FFT_2(cufftComplex *v)
{
    cufftComplex v0 = v[0];
    v[0] = cuCaddf(v0, v[1]);
    v[1] = cuCsubf(v0, v[1]);
}

void cpu_FFT(cufftComplex *v)
{
    if (R == 2)
    {
        cpu_FFT_2(v);
    }
}

void cpu_FfIteration(long j, long N, long Ns, cufftComplex *data0, cufftComplex *data1)
{
    cufftComplex v[R];
    long idxS = j;
    float angle = -2 * M_PI * (j % Ns) / (Ns * R);
    for (int r = 0; r < R; r++)
    {
        v[r] = data0[idxS + r * N / R];
        v[r] = cuCmulf(v[r], make_cuFloatComplex(cos(r * angle), sin(r * angle)));
    }
    cpu_FFT(v);
    long idxD = cpu_expand(j, Ns, R);
    for (int r = 0; r < R; r++)
    {
        data1[idxD + r * Ns] = v[r];
    }
}

cufftComplex *CPU_FFT(long N, cufftComplex *data0, cufftComplex *data1)
{
    for (long Ns = 1; Ns < N; Ns *= R)
    {
        for (long j = 0; j < N / R; j++)
        {
            cpu_FfIteration(j, N, Ns, data0, data1);
        }
        cpu_swap(data0, data1);
    }
    return data0;
}

int main(int argc, char **argv)
{
    runTest();
    return 0;
}

void runTest()
{
    cout << "runTest is starting..." << endl;
    cout << "Radix is " << R << endl;
    cout << "Input size is " << INPUT_SIZE << endl;

    long input_size = INPUT_SIZE;
    long mem_size = sizeof(cufftComplex) * input_size;

#ifdef USE_CUDA_MALLOC_MANAGED
    cufftComplex *d_idata1;
    cufftComplex *d_idata2;
    cufftComplex *d_odata1;
    cufftComplex *d_odata2;
    cudaMallocManaged(&d_idata1, mem_size);
    cudaMallocManaged(&d_idata2, mem_size);
    cudaMallocManaged(&d_odata1, mem_size);
    cudaMallocManaged(&d_odata2, mem_size);

    for (unsigned long i = 0; i < input_size; ++i)
    {
        d_idata1[i].x = rand() / (float)RAND_MAX;
        d_idata1[i].y = 0;
        d_idata2[i].x = d_idata1[i].x;
        d_idata2[i].y = 0;
        d_odata1[i].x = 0;
        d_odata1[i].y = 0;
        d_odata2[i].x = 0;
        d_odata2[i].y = 0;
    }

#else

    // Allocate host memory for the input
    cufftComplex *h_idata1 = (cufftComplex *)malloc(mem_size);
    cufftComplex *h_idata2 = (cufftComplex *)malloc(mem_size);
    cufftComplex *h_odata1 = (cufftComplex *)malloc(mem_size);
    cufftComplex *h_odata2 = (cufftComplex *)malloc(mem_size);
    // Initalize the memory for the input
    for (unsigned long i = 0; i < input_size; ++i)
    {
        h_idata1[i].x = rand() / (float)RAND_MAX;
        h_idata1[i].y = 0;
        h_idata2[i].x = h_idata1[i].x;
        h_idata2[i].y = 0;
        h_odata1[i].x = 0;
        h_odata1[i].y = 0;
        h_odata2[i].x = 0;
        h_odata2[i].y = 0;
    }

    cufftComplex *d_idata1;
    cufftComplex *d_odata1;
    cufftComplex *d_idata2;
    cufftComplex *d_odata2;

    // Allocate device memory
    checkCuda(cudaMalloc(&d_idata1, mem_size), 508);
    checkCuda(cudaMalloc(&d_idata2, mem_size), 509);
    // Copy host memory to device
    checkCuda(cudaMemcpy(d_idata1, h_idata1, mem_size,
                         cudaMemcpyHostToDevice),
              511);
    checkCuda(cudaMemcpy(d_idata2, d_idata1, mem_size,
                         cudaMemcpyDeviceToDevice),
              513);

    // Allocate device memory
    checkCuda(cudaMalloc(&d_odata1, mem_size), 517);
    checkCuda(cudaMalloc(&d_odata2, mem_size), 518);
    // Copy host memory to device
    checkCuda(cudaMemcpy(d_odata1, h_odata1, mem_size,
                         cudaMemcpyHostToDevice),
              520);
    checkCuda(cudaMemcpy(d_odata2, d_odata1, mem_size,
                         cudaMemcpyDeviceToDevice),
              522);
#endif

    auto start = chrono::high_resolution_clock::now();

    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, input_size, CUFFT_C2C, 1);

    // CUFTT exec
    cufftExecC2C(plan, (cufftComplex *)d_idata1, (cufftComplex *)d_odata1, CUFFT_FORWARD);

    auto end = chrono::high_resolution_clock::now();

    auto microseconds = chrono::duration_cast<std::chrono::microseconds>(end - start);

    cout << "cufft elapsed  " << microseconds.count() << "us" << endl;
    // Destroy CUFFT context
    cufftDestroy(plan);

    start = chrono::high_resolution_clock::now();

    GPU_FFT(input_size, d_idata2, d_odata2);
    // GPU_FftShMem(input_size, d_idata2, d_odata2); // not work
    // h_odata2 = CPU_FFT(input_size, h_idata2, h_odata2);
    cudaDeviceSynchronize();

    end = chrono::high_resolution_clock::now();

    microseconds = chrono::duration_cast<std::chrono::microseconds>(end - start);

    cout << "my fft elapsed " << microseconds.count() << "us" << endl;

#ifdef USE_CUDA_MALLOC_MANAGED
    // d_idata
    cout << d_odata1[1].x << endl;
    cout << d_odata2[1].x << endl;
    cout << l2err(input_size, d_odata1, d_odata2) << endl;
#else
    checkCuda(cudaMemcpy(h_odata1, d_odata1, mem_size,
                         cudaMemcpyDeviceToHost),
              558);
    checkCuda(cudaMemcpy(h_odata2, d_odata2, mem_size,
                         cudaMemcpyDeviceToHost),
              560);
    cout << h_odata1[1].x << endl;
    cout << h_odata2[1].x << endl;
    cout << l2err(input_size, h_odata1, h_odata2) << endl;
#endif

#ifdef USE_CUDA_MALLOC_MANAGED
    checkCuda(cudaFree(d_idata1), 575);
    checkCuda(cudaFree(d_odata1), 576);
    checkCuda(cudaFree(d_idata2), 577);
    checkCuda(cudaFree(d_odata2), 578);
#else
    free(h_idata1);
    free(h_idata2);
    free(h_odata1);
    free(h_odata2);
    checkCuda(cudaFree(d_idata1), 575);
    checkCuda(cudaFree(d_odata1), 576);
    checkCuda(cudaFree(d_idata2), 577);
    checkCuda(cudaFree(d_odata2), 578);
#endif
}