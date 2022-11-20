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

/*
__global__ void sieve(bool* numberArray, __int64 k) {
	__int64 i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (i >= k + 1) {
		i += blockDim.x * gridDim.x;
		__syncthreads();
	}
	__int64 j = i;
	while (i + j + 2 * i * j <= k)
	{
		numberArray[i + j + 2 * i * j] = false;
		j++;
	}
}
*/

__global__ void sieve(bool* numberArray, __int64 k) {
	__int64 i = blockIdx.x * blockDim.x + threadIdx.x + 1;

	if (i < k + 1) {
		__int64 j = i;
		while (i + j + 2 * i * j <= k)
		{
			numberArray[i + j + 2 * i * j] = false;

			j++;
		}
	}
}

int main(int argc, char** argv) {
	int devID;
	cudaDeviceProp props;

	devID = findCudaDevice(argc, (const char**)argv);
	checkCudaErrors(cudaGetDevice(&devID));
	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
	printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name,
		props.major, props.minor);

	int n = 100000000;
	int k = (n - 2) / 2;
	bool* boolArray = new bool[k + 1];
	for (int i = 0; i < k + 1; i++)
		boolArray[i] = true;

	bool* d_boolArrray;
	cudaMalloc(&d_boolArrray, (k + 1) * sizeof(bool));
	cudaMemcpy(d_boolArrray, boolArray, (k + 1) * sizeof(bool), cudaMemcpyHostToDevice);


	clock_t t;
	t = clock();

	int numBlocks = ceil((double)(k + 1) / 1024);
	sieve << <numBlocks, 1024 >> > (d_boolArrray, k);

	cudaDeviceSynchronize();

	t = clock() - t;
	printf("T = %d, Time = %lf\n", 1024, (((double)t) / CLOCKS_PER_SEC) * 1000);

	cudaMemcpy(boolArray, d_boolArrray, (k + 1) * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(d_boolArrray);
	int count = 0;
	if (n > 2)
	{
		// printf("%d ", 2);
		count++;
	}
	for (int i = 1; i < k + 1; i++)
	{
		if (boolArray[i])
		{
			// printf("%d ", 2 * i + 1);
			count++;
		}
	}
	printf("Count: %d\n", count);


	delete[] boolArray;

	return EXIT_SUCCESS;
}
