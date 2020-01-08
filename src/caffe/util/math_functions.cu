#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>
#include <cuda_runtime.h>
#define TILE_WIDTH 16
#define BLOCK_SIZE 16 
namespace caffe {

  //gemv-------------------------------------------------------------------------------------------------------------------------------------------
  template<typename T>
  __global__ void matvec_kernel_ILP2(const T * __restrict__ dA, const T * __restrict__ dx, T * __restrict__ dy, const unsigned int nRows, const unsigned int nCols, const T alpha, const bool trans)
  {
    // LOG(INFO) << ("Warm hug from my kernle. ln 19.\n");
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
    // printf("%d %d %d %d\n",  tid, threadIdx.x, blockIdx.x, blockDim.x);
    // printf("sdfasdf %d\n", gridDim.x * BLOCK_SIZE);
    // printf("%d\t%d\t%d\t%d\t\n", blockIdx.x, blockIdx.y,threadIdx.x,threadIdx.y);
  
    __shared__ T x_shared[BLOCK_SIZE];
  
    T y_val1 = 0.0;
    T y_val2 = 0.0;
  
    int tmp;
    if (trans) {tmp = ((nRows + BLOCK_SIZE - 1)/ BLOCK_SIZE);}
    else {tmp = ((nCols + BLOCK_SIZE - 1)/ BLOCK_SIZE);  }
  
    #pragma unroll
    for (unsigned int m = 0; m < tmp; ++m)
    {
      if (trans){
        if ((m * BLOCK_SIZE + threadIdx.x) <  nRows) 
          x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
        else
          x_shared[threadIdx.x] = 0.f;
      }
      else{
        if ((m * BLOCK_SIZE + threadIdx.x) <  nCols) 
          x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
        else
          x_shared[threadIdx.x] = 0.f;
      }
      
      __syncthreads();
  
      #pragma unroll
      for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
        if (trans){
          if ((tid) * nRows + (e + BLOCK_SIZE * m) < nRows*nCols){
            y_val1 += dA[(tid) * nRows + (e + BLOCK_SIZE * m)] * x_shared[e];
          }
          if ((tid + gridDim.x * BLOCK_SIZE) * nRows + (e + BLOCK_SIZE * m) < nRows*nCols){
            y_val2 += dA[(tid + gridDim.x * BLOCK_SIZE) * nRows + (e + BLOCK_SIZE * m)] * x_shared[e];
          }
        }
        else{
          if (tid + (e + BLOCK_SIZE * m) * nRows < nRows*nCols){
            y_val1 += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
          }
          if (tid + gridDim.x * BLOCK_SIZE + (e + BLOCK_SIZE * m) * nRows < nRows*nCols){
            y_val2 += dA[tid + gridDim.x * BLOCK_SIZE + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
          } 
        }
      }
  
      __syncthreads();
    }
    if (trans){
      if (tid < nCols) dy[tid] = y_val1 * alpha;
      if ((tid + gridDim.x * BLOCK_SIZE) < nCols) dy[tid + gridDim.x * BLOCK_SIZE] = y_val2 * alpha;
    }
    else{
      if (tid < nRows) dy[tid] = y_val1 * alpha;
      if ((tid + gridDim.x * BLOCK_SIZE) < nRows) dy[tid + gridDim.x * BLOCK_SIZE] = y_val2 * alpha;
    }
  }
  
  template <>
  void kk_gpu_gemv<float>(const float* h_A, const float* h_x, float* h_y, const unsigned int nRows, const unsigned int nCols, const float alpha, const bool trans) {
    int size = sizeof(float);
    // LOG(INFO) << ("Warm hug from my func. ln 51.\n");
    // LOG(INFO) << ("My caffe_gpu_gemv invoked in math_functions.cu.\n");
    float *d_A;
    float *d_x;
    float *d_y;
    if (trans){
      cudaMalloc((void**)&d_A, nRows*nCols*size);
      cudaMalloc((void**)&d_x, nRows*size);
      cudaMalloc((void**)&d_y, nCols*size);
      cudaMemcpy(d_A, h_A, nRows*nCols*size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_x, h_x, nRows*size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, h_y, nCols*size, cudaMemcpyHostToDevice);
    }
    else{
      cudaMalloc((void**)&d_A, nRows*nCols*size);
      cudaMalloc((void**)&d_x, nCols*size);
      cudaMalloc((void**)&d_y, nRows*size);
      cudaMemcpy(d_A, h_A, nRows*nCols*size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_x, h_x, nCols*size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, h_y, nRows*size, cudaMemcpyHostToDevice);
    }
    
    if (trans){
      dim3 dim_grid(((nCols+1)/2 + BLOCK_SIZE -1)/ BLOCK_SIZE);
      // printf("grid size %d\n", (nRows/2 + BLOCK_SIZE -1)/ BLOCK_SIZE);
      dim3 dim_block(BLOCK_SIZE);
      matvec_kernel_ILP2<float> <<<dim_grid, dim_block>>>(d_A, d_x, d_y, nRows, nCols, alpha, trans);
      cudaMemcpy(h_y, d_y, nCols*size, cudaMemcpyDeviceToHost);
    }
    else{
      dim3 dim_grid(((nRows+1)/2 + BLOCK_SIZE -1)/ BLOCK_SIZE);
      // printf("grid size %d\n", (nRows/2 + BLOCK_SIZE -1)/ BLOCK_SIZE);
      dim3 dim_block(BLOCK_SIZE); 
      matvec_kernel_ILP2<float> <<<dim_grid, dim_block>>>(d_A, d_x, d_y, nRows, nCols, alpha, trans);
      cudaMemcpy(h_y, d_y, nRows*size, cudaMemcpyDeviceToHost);
    }
    // printf("block_size %d\n", BLOCK_SIZE);
    // if (trans){
    //   matvec_kernel_ILP2<float> <<<dim_grid, dim_block>>>(d_A, d_x, d_y, nCols, nRows, alpha);
    //   cudaMemcpy(h_y, d_y, nCols*size, cudaMemcpyDeviceToHost);
    // }
    // else{
    // }
  
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
  }
  
  template <>
  void kk_gpu_gemv<double>(const double* h_A, const double* h_x, double* h_y, const unsigned int nRows, const unsigned int nCols, const double alpha, const bool trans) {
    int size = sizeof(double);
    // LOG(INFO) << ("Warm hug from my func. ln 51.\n");
    // LOG(INFO) << ("My caffe_gpu_gemv invoked in math_functions.cu.\n");
    double *d_A;
    double *d_x;
    double *d_y;
    if (trans){
      cudaMalloc((void**)&d_A, nRows*nCols*size);
      cudaMalloc((void**)&d_x, nRows*size);
      cudaMalloc((void**)&d_y, nCols*size);
      cudaMemcpy(d_A, h_A, nRows*nCols*size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_x, h_x, nRows*size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, h_y, nCols*size, cudaMemcpyHostToDevice);
    }
    else{
      cudaMalloc((void**)&d_A, nRows*nCols*size);
      cudaMalloc((void**)&d_x, nCols*size);
      cudaMalloc((void**)&d_y, nRows*size);
      cudaMemcpy(d_A, h_A, nRows*nCols*size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_x, h_x, nCols*size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, h_y, nRows*size, cudaMemcpyHostToDevice);
    }
    
    if (trans){
      dim3 dim_grid(((nCols+1)/2 + BLOCK_SIZE -1)/ BLOCK_SIZE);
      // printf("grid size %d\n", (nRows/2 + BLOCK_SIZE -1)/ BLOCK_SIZE);
      dim3 dim_block(BLOCK_SIZE);
      matvec_kernel_ILP2<double> <<<dim_grid, dim_block>>>(d_A, d_x, d_y, nRows, nCols, alpha, trans);
      cudaMemcpy(h_y, d_y, nCols*size, cudaMemcpyDeviceToHost);
    }
    else{
      dim3 dim_grid(((nRows+1)/2 + BLOCK_SIZE -1)/ BLOCK_SIZE);
      // printf("grid size %d\n", (nRows/2 + BLOCK_SIZE -1)/ BLOCK_SIZE);
      dim3 dim_block(BLOCK_SIZE); 
      matvec_kernel_ILP2<double> <<<dim_grid, dim_block>>>(d_A, d_x, d_y, nRows, nCols, alpha, trans);
      cudaMemcpy(h_y, d_y, nRows*size, cudaMemcpyDeviceToHost);
    }
    // printf("block_size %d\n", BLOCK_SIZE);
    // if (trans){
    //   matvec_kernel_ILP2<float> <<<dim_grid, dim_block>>>(d_A, d_x, d_y, nCols, nRows, alpha);
    //   cudaMemcpy(h_y, d_y, nCols*size, cudaMemcpyDeviceToHost);
    // }
    // else{
    // }
  
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
  }
  //gemv-------------------------------------------------------------------------------------------------------------------------------------------
  
//axpby
template <typename T>
__global__ void axpby_kernel(const int N, const T alpha, const T beta, const T*  X, T*  Y){
    CUDA_KERNEL_LOOP(i, N){
      // if (i < N)
        Y[i] = X[i]*alpha + Y[i]*beta;
    }
}

template <>
void kk_gpu_axpby<float>(const int N, const float alpha, const float beta, const float* X, float* Y){
    float *d_X, *d_Y;
    size_t size = N*sizeof(float);
    cudaMalloc((void**)&d_X, size);
    cudaMalloc((void**)&d_Y, size);
    cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, size, cudaMemcpyHostToDevice);
    dim3 dim_grid((N+BLOCK_SIZE)/BLOCK_SIZE);
    dim3 dim_block(BLOCK_SIZE);
    axpby_kernel<float> <<<dim_grid, dim_block>>>(N, alpha, beta, d_X, d_Y);
    cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_Y);
}
template <>
void kk_gpu_axpby<double>(const int N, const double alpha, const double beta, const double* X, double* Y){
  double *d_X, *d_Y;
    size_t size = N*sizeof(double);
    cudaMalloc((void**)&d_X, size);
    cudaMalloc((void**)&d_Y, size);
    cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, size, cudaMemcpyHostToDevice);
    dim3 dim_grid((N+BLOCK_SIZE)/BLOCK_SIZE);
    dim3 dim_block(BLOCK_SIZE);
    axpby_kernel<double> <<<dim_grid, dim_block>>>(N, alpha, beta, d_X, d_Y);
    cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_Y);
}
//axpby

//scale
template <typename T>
__global__ void scale_kernel(const int N, const T alpha, const T*  X, T* Y){
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    CUDA_KERNEL_LOOP(i, N){
      // if (i < N)
        Y[i] = X[i]*alpha;
    }
}

template <>
void kk_gpu_scale<float>(const int N, const float alpha, const float* X, float* Y){
    float *d_X, *d_Y;
    size_t size = N*sizeof(float);
    cudaMalloc((void**)&d_X, size);
    cudaMalloc((void**)&d_Y, size);
    cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
    dim3 dim_grid((N+BLOCK_SIZE)/BLOCK_SIZE);
    dim3 dim_block(BLOCK_SIZE);
    scale_kernel<float> <<<dim_grid, dim_block>>>(N, alpha, d_X, d_Y);
    cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_Y);
}
template <>
void kk_gpu_scale<double>(const int N, const double alpha, const double* X, double* Y){
    double *d_X, *d_Y;
    size_t size = N*sizeof(double);
    cudaMalloc((void**)&d_X, size);
    cudaMalloc((void**)&d_Y, size);
    cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
    dim3 dim_grid((N+BLOCK_SIZE)/BLOCK_SIZE);
    dim3 dim_block(BLOCK_SIZE);
    scale_kernel<double> <<<dim_grid, dim_block>>>(N, alpha, d_X, d_Y);
    cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_Y);
}
//scale

//axpy
template <typename T>
__global__ void axpy_kernel(const int N, const T alpha, const T*  X, T*  Y){
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    CUDA_KERNEL_LOOP(i, N){
      // if (i < N)
        Y[i] = X[i]*alpha + Y[i];
    }
}

template <>
void kk_gpu_axpy<float>(const int N, const float alpha, const float* X, float* Y){
    float *d_X, *d_Y;
    size_t size = N*sizeof(float);
    cudaMalloc((void**)&d_X, size);
    cudaMalloc((void**)&d_Y, size);
    cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, size, cudaMemcpyHostToDevice);
    dim3 dim_grid((N+BLOCK_SIZE)/BLOCK_SIZE);
    dim3 dim_block(BLOCK_SIZE);
    axpy_kernel<float> <<<dim_grid, dim_block>>>(N, alpha, d_X, d_Y);
    cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_Y);
}
template <>
void kk_gpu_axpy<double>(const int N, const double alpha, const double* X, double* Y){
    double *d_X, *d_Y;
    size_t size = N*sizeof(double);
    cudaMalloc((void**)&d_X, size);
    cudaMalloc((void**)&d_Y, size);
    cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, size, cudaMemcpyHostToDevice);
    dim3 dim_grid((N+BLOCK_SIZE)/BLOCK_SIZE);
    dim3 dim_block(BLOCK_SIZE);
    axpy_kernel<double> <<<dim_grid, dim_block>>>(N, alpha, d_X, d_Y);
    cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_Y);
}
//axpy

//dot
template <typename T>
__global__ void dot_kernel(const int N, const T* X, const T* Y, T* res){
    CUDA_KERNEL_LOOP(i, N){
      // if (i < N)
        res[i] = X[i]*Y[i];
    }
}

template <>
void kk_gpu_dot<float>(const int N, const float* X, const float* Y, float* res){
    float *d_X, *d_Y, *d_res;
    size_t size = N*sizeof(float);
    cudaMalloc((void**)&d_X, size);
    cudaMalloc((void**)&d_Y, size);
    cudaMalloc((void**)&d_res, size);
    cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, size, cudaMemcpyHostToDevice);
    dim3 dim_grid((N+BLOCK_SIZE)/BLOCK_SIZE);
    dim3 dim_block(BLOCK_SIZE);
    dot_kernel<float> <<<dim_grid, dim_block>>>(N, d_X, d_Y, d_res);
    cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_res);
}
template <>
void kk_gpu_dot<double>(const int N, const double* X, const double* Y, double* res){
  double *d_X, *d_Y, *d_res;
    size_t size = N*sizeof(double);
    cudaMalloc((void**)&d_X, size);
    cudaMalloc((void**)&d_Y, size);
    cudaMalloc((void**)&d_res, size);
    cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, size, cudaMemcpyHostToDevice);
    dim3 dim_grid((N+BLOCK_SIZE)/BLOCK_SIZE);
    dim3 dim_block(BLOCK_SIZE);
    dot_kernel<double> <<<dim_grid, dim_block>>>(N, d_X, d_Y, d_res);
    cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_res);
}
//dot

















using namespace std;
template<typename Dtype>
__global__ void MatrixMulKernle(int m, int n, int k, Dtype *A,Dtype  *B, Dtype *C,bool IsAdd_C,bool TransA,bool TransB)
{
    //申请共享内存，存在于每个block中
  __shared__ Dtype ds_A[TILE_WIDTH][TILE_WIDTH]; 
  __shared__ Dtype ds_B[TILE_WIDTH][TILE_WIDTH];
  
  //简化坐标记法,出现下面6个表示的地方就是并行的地方。
  int bx = blockIdx.x;		int by = blockIdx.y;
  int tx = threadIdx.x;		int ty = threadIdx.y;
  
  //确定结果矩阵中的行和列
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  
  //临时变量
  Dtype Cvalue=0;

  
  
  //循环读入A,B瓦片，计算结果矩阵，分阶段进行计算
  for (int t=0; t<(n-1)/TILE_WIDTH+1; ++t)
  {
    //将A,B矩阵瓦片化的结果放入shared memory中，每个线程加载相应于C元素的A/B矩阵元素
    if (Row < m && t * TILE_WIDTH + tx < n)		//越界处理，满足任意大小的矩阵相乘（可选）
      //ds_A[tx][ty] = A[t*TILE_WIDTH + tx][Row];
      if (TransA==false){
        ds_A[ty][tx] = A[Row*n+t*TILE_WIDTH+tx];//以合并的方式加载瓦片
        // printf("no transA");
      }
      else{
        // printf("uiou");
        ds_A[ty][tx] = A[(t*TILE_WIDTH+tx)*m+Row];//trans A[Row*n+t*TILE_WIDTH+tx];//以合并的方式加载瓦片,tx列，ty行

      }
        
    else
      ds_A[ty][tx] = 0.0;
  
    if (t * TILE_WIDTH + ty < n && Col < k)
      //ds_B[tx][ty] = B[Col][t*TILE_WIDTH + ty];
      if (TransB==false){
        ds_B[ty][tx] = B[(t*TILE_WIDTH + ty)*k+Col];

      }
      else{
        // printf("B");
        // ds_B[ty][tx] = B[(t*TILE_WIDTH + ty)*k+Col];
        // printf("%f ",ds_B[ty][tx]);
        ds_B[ty][tx] = B[(t*TILE_WIDTH + ty)+Col*n];
        // printf("%f\n",ds_B[ty][tx]);
       

      }
    else
      ds_B[ty][tx] = 0.0;	
  
    //保证tile中所有的元素被加载
    __syncthreads();
    
    if (IsAdd_C==0){
      Cvalue+=(ds_A[ty][0]*ds_B[0][tx]+ds_A[ty][1]*ds_B[1][tx]+ds_A[ty][2]*ds_B[2][tx]+ds_A[ty][3]*ds_B[3][tx]+ \
        ds_A[ty][4]*ds_B[4][tx]+ds_A[ty][5]*ds_B[5][tx]+ds_A[ty][6]*ds_B[6][tx]+ds_A[ty][7]*ds_B[7][tx]+ \
        ds_A[ty][8]*ds_B[8][tx]+ds_A[ty][9]*ds_B[9][tx]+ds_A[ty][10]*ds_B[10][tx]+ds_A[ty][11]*ds_B[11][tx]+ \
        ds_A[ty][12]*ds_B[12][tx]+ds_A[ty][13]*ds_B[13][tx]+ds_A[ty][14]*ds_B[14][tx]+ds_A[ty][15]*ds_B[15][tx]);
    }
    else{
      C[k*Row+Col]+=(ds_A[ty][0]*ds_B[0][tx]+ds_A[ty][1]*ds_B[1][tx]+ds_A[ty][2]*ds_B[2][tx]+ds_A[ty][3]*ds_B[3][tx]+ \
        ds_A[ty][4]*ds_B[4][tx]+ds_A[ty][5]*ds_B[5][tx]+ds_A[ty][6]*ds_B[6][tx]+ds_A[ty][7]*ds_B[7][tx]+ \
        ds_A[ty][8]*ds_B[8][tx]+ds_A[ty][9]*ds_B[9][tx]+ds_A[ty][10]*ds_B[10][tx]+ds_A[ty][11]*ds_B[11][tx]+ \
        ds_A[ty][12]*ds_B[12][tx]+ds_A[ty][13]*ds_B[13][tx]+ds_A[ty][14]*ds_B[14][tx]+ds_A[ty][15]*ds_B[15][tx]);
    }

    __syncthreads();
    if (IsAdd_C==0){
      if(Row < m && Col < k){
        // C[Col*m+Row]=Cvalue;
        C[k*Row+Col]=Cvalue;

      }
    }

    // for (int i = 0; i < TILE_WIDTH; ++i){
    //   if (IsAdd_C==0){
    //     Cvalue += ds_A[ty][i] * ds_B[i][tx];//从shared memory中取值

    //   }
    //   else{
    //     if(Row < m && Col < k)
    //       // C[Col*m+Row]+= ds_A[ty][i] * ds_B[i][tx];
    //       C[k*Row+Col]+= ds_A[ty][i] * ds_B[i][tx];

    //   }
    // }
    
  
    // //确保所有线程完成计算后，进行下一个阶段的计算
    // __syncthreads();
    // if (IsAdd_C==0){
    //   if(Row < m && Col < k){
    //     // C[Col*m+Row]=Cvalue;
    //     C[k*Row+Col]=Cvalue;

    //   }
    // }





    // if(Row < m && Col < k){
    //   if (IsAdd_C==0){
    //     C[Col*m+Row]=Cvalue;
    //   }
    //   else{
    //     printf("%d %d %f %f\n",Col,Row,C[Col*m+Row],Cvalue);
    //     C[Col*m+Row]=Cvalue;
    //     // printf("wyiyiyyyi\n");
    //   }
    // }

      

  }
}
  



template <>
void caffe_gpu_gemm<float>(const int m, const int n, const int k,
     const float* A, const float* B,   float* C,bool IsAdd_C,bool TransA,bool TransB) {
  // Note that cublas follows fortran order.
  // printf("2444444333333333\n");

  //lzy
  //分配显存空间
  int size = sizeof(float);
  float *d_a;
  float *d_b;
  float *d_c;
  cudaMalloc((void**)&d_a,m*n*size);
  cudaMalloc((void**)&d_b,n*k*size);
  cudaMalloc((void**)&d_c,m*k*size);

  //把数据从Host传到Device
  cudaMemcpy(d_a, A, size*m*n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, size*n*k, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, C, size*m*k, cudaMemcpyHostToDevice);


  //分配网格结构
  int tile_width=TILE_WIDTH;
  dim3 dimGrid((k-1)/tile_width+1,(m-1)/tile_width+1,1);	//向上取整
  dim3 dimBlock(tile_width,tile_width,1);
  //lzy
  MatrixMulKernle<float><<<dimGrid,dimBlock>>>(m,n,k,d_a,d_b,d_c,IsAdd_C,TransA,TransB);
  // MatrixMulKernle<float><<<dimGrid,dimBlock>>>(m,k,n,d_b,d_a,d_c,IsAdd_C,TransA,TransB);

  cudaMemcpy(C, d_c, size*m*k, cudaMemcpyDeviceToHost);

	cudaFree(d_c);
  cudaFree(d_a);
	cudaFree(d_b);
}

template <>
void caffe_gpu_gemm<double>(const int m, const int n, const int k,
     const double* A, const double* B,   double* C,bool IsAdd_C,bool TransA,bool TransB) {
  // Note that cublas follows fortran order.
  // printf("24444433333333\n");

  //分配显存空间
  int size = sizeof(double);
  double *d_a;
  double *d_b;
  double *d_c;
  cudaMalloc((void**)&d_a,m*n*size);
  cudaMalloc((void**)&d_b,n*k*size);
  cudaMalloc((void**)&d_c,m*k*size);

  //把数据从Host传到Device
  cudaMemcpy(d_a, A, size*m*n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, size*n*k, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, C, size*m*k, cudaMemcpyHostToDevice);


  //分配网格结构
  int tile_width=TILE_WIDTH;
  dim3 dimGrid((k-1)/tile_width+1,(m-1)/tile_width+1,1);	//向上取整
  dim3 dimBlock(tile_width,tile_width,1);
  //lzy
  MatrixMulKernle<double><<<dimGrid,dimBlock>>>(m,n,k,d_a,d_b,d_c,IsAdd_C,TransA,TransB);
  // MatrixMulKernle<double><<<dimGrid,dimBlock>>>(k,n,m,d_b,d_a,d_c,IsAdd_C,TransA,TransB);
// 
  cudaMemcpy(C, d_c, size*m*k, cudaMemcpyDeviceToHost);

	cudaFree(d_c);
  cudaFree(d_a);
	cudaFree(d_b);
}









template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  // printf("2333333333333333333\n");
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
      // LOG(INFO) << ("caffe_gpu_gemv invoked in math_functions.cu.\n");
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
