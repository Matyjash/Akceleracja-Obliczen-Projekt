/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cmath>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

__global__ void sieve(bool* numberArray, __int64 series_num, __int64 k) {
    // __int64 i = blockDim.x * series_num + threadIdx.x;
    __int64 i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    // printf("Block: %d, seria: %d, id: %d\n", blockDim.x, series_num, threadIdx.x);
    if (i >= k + 1) {
        return;
    }
    __int64 j = i;
    while (i + j + 2 * i * j <= k)
    {
        //je¿eli mamy liczbê nieparzyst¹ w postaci 2k+1 (uzyskujemy j¹ przy wypisywaniu wyniku) 
        //to je¿eli liczba ta ma postaæ i+j+2ij to mo¿emy j¹ wykluczyæ ze zbioru liczb nieparzystych
        numberArray[i + j + 2 * i * j] = false;
        // printf("%d - false, thread: i: %d, id: %d, T: %d\n", 2 * (i + j + 2 * i * j) + 1, i, threadIdx.x, blockDim.x);
        j++;
    }
}

int main(int argc, char **argv) {
  int devID;
  cudaDeviceProp props;

  // This will pick the best possible CUDA capable device
  devID = findCudaDevice(argc, (const char **)argv);

  // Get GPU information
  checkCudaErrors(cudaGetDevice(&devID));
  checkCudaErrors(cudaGetDeviceProperties(&props, devID));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name,
         props.major, props.minor);

  // Kernel configuration, where a two-dimensional grid and
  // three-dimensional blocks are configured.
  // __int64 n = 9223372036854775807;
  __int64 n = 100000000;
  __int64 k = (n - 2) / 2;
  // int N = 1 << 27;//1.34217728 *10^8 elements. 512 MB
  bool* boolArray;
  cudaMallocManaged(&boolArray, (k + 1) * sizeof(bool));

  for (int i = 0; i < k + 1; i++)
      boolArray[i] = true;

  clock_t t;


  for (int thr = 1024; thr <= 1024 * 3; thr += 1024)
  {

   // int T = max(thr, 1); // we will need atleast 1 thread.
  int T = 1024;
    t = clock();
    /*
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(T, 1, 1);

    for (__int64 i = 0; i < (k + 1) / T + 1; i++)
    sieve<<<dimGrid, dimBlock>>>(boolArray, i, k);
    */

    // for (__int64 i = 0; i < (k + 1) / T + 1; i++)
    int numBlocks = ((k + 1) / T + 1) * 64;
    dim3 threadsPerBlock(T / 64);
    sieve<<<numBlocks, threadsPerBlock>>>(boolArray, 0, k);
    // TODO: upewniæ siê ¿e jeœli (k + 1) nie jest podzielne przez T to ostatnie iteracje s¹ wykonywane tak jak powinny

    cudaDeviceSynchronize();

    t = clock() - t;
    printf("T = %d, Time = %lf\n", T, (((double)t) / CLOCKS_PER_SEC) * 1000);
      
    int count = 0;
    if (n > 2)
    {
        // printf("%d ", 2);
    }
    for (int i = 1; i < k + 1; i++)
    {
        if (boolArray[i])
        {
            // liczba w postaci 2k + 1
            // printf("%d ", 2 * i + 1);
            count++;
        }
    }
    printf("Count: %d\n", count);

}

  return EXIT_SUCCESS;
}
