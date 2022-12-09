#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "scene.cuh"
#include <iostream>
#include <stdio.h>
#include <SFML/Graphics.hpp>
#include <chrono>
#include <string>



__device__ void kernelCalculatePos(Particle& p, const float dt)
{
	const float vel_x = p.curPos.x - p.prevPos.x, vel_y = p.curPos.y - p.prevPos.y;
	p.prevPos.x = p.curPos.x;
	p.prevPos.y = p.curPos.y;
	const float m = sqrtf(vel_x * vel_x + vel_y * vel_y);
	const float mc = p.pressure;
	const float pc = 1.f / (1.f + mc);

	p.curPos.x += (vel_x / (1.f + mc / 10.f) + (p.acceleration.x * dt * dt * pc * pc));
	p.curPos.y += (vel_y / (1.f + mc / 10.f) + (p.acceleration.y * dt * dt * pc * pc));

	p.pressureCoef = pc;
	p.pressure = 0.f;
}

__device__ void kernelAccelerate(Particle& p, sf::Vector2f a)
{
	p.acceleration.x += a.x;
	p.acceleration.y += a.y;
}

__device__ void kernelApplyConstraint(Particle& p, sf::Vector2f pos1, sf::Vector2f pos2)
{
	const float
		x = p.curPos.x,
		y = p.curPos.y,
		r = p.radius;
	if (x + r > pos2.x)
		p.curPos.x = pos2.x - r;
	if (y + r > pos2.y)
		p.curPos.y = pos2.y - r;
	if (x - r < pos1.x)
		p.curPos.x = pos1.x + r;
	if (y - r < pos1.y)
		p.curPos.y = pos1.y + r;
}

__device__ void kernelCollide(Particle& p1, Particle& p2)
{
	const float ax = p1.curPos.x - p2.curPos.x;
	const float ay = p1.curPos.y - p2.curPos.y;
	const float ndist = (ax * ax) + (ay * ay);
	const float r2 = p1.radius + p2.radius;
	if (ndist > r2 * r2)
		return;
	const float dist = sqrtf(ndist); 
	if (!dist)
		return;
	const float delta = ((r2 - dist)) * 0.5f;
	const float nx = ax / dist * delta;
	const float ny = ay / dist * delta;
	p1.curPos.x += nx;
	p1.curPos.y += ny;
	p2.curPos.x -= nx;
	p2.curPos.y -= ny;
	p1.pressure += delta / 2.f;
	p2.pressure += delta / 2.f;
}

 
__device__ void staticGridCollide(cui pos1, cui pos2, int* grid, int* cellCount, Particle* particles, cui gridWidth, cui gridHeight, cui cellSize)
{
	if (pos2 < 0 || pos2 >= gridWidth * gridHeight)
		return;
	for (int i{}; i < cellCount[pos1]; ++i)
		for (int j = i; j < cellCount[pos2]; ++j)
		{
			kernelCollide(particles[grid[pos1 * cellSize + i]], particles[grid[pos2 * cellSize + j]]);
		}	
}

__global__ void calculateCollisions(int* grid, int* cellCount, Particle* particles, cui gridWidth, cui gridHeight, cui cellSize, float radius, cui count, cui delta)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + delta;
	if (i >= count) return;
	// �������� ����������
	int x = (int)(particles[i].curPos.x / (radius * 2.f));
	int y = (int)(particles[i].curPos.y / (radius * 2.f));
	// �������� ���������
	if (x < 0)
		x = 0;
	if (y < 0)
		y = 0;
	if (x >= gridWidth)
		x = gridWidth - 1;
	if (y >= gridHeight)
		y = gridHeight - 1;
	int pos2 = x * gridHeight + y;
	// ��-����, ����� ��������� ������ ���� ����������, ��� �� �������� � ������������ ����������
	if (pos2 != i && cellCount[pos2] < cellSize)
	{
		int old = atomicAdd(&cellCount[pos2], 1);
		grid[pos2 * cellSize + old] = i;
	}
}

__global__ void correctGrid(int* cellCount, cui gridWidth, cui gridHeight, cui cellSize, cui delta)
{
	cui i = blockIdx.x * blockDim.x + threadIdx.x + delta;
	if (i >= gridHeight * gridWidth) return;
	if (cellCount[i] > cellSize) cellCount[i] = cellSize;
}

__global__ void applyGravityKernel(Particle* particles, cui count, sf::Vector2f g)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count) return;
	kernelAccelerate(particles[i], g);
}

__global__ void applyCollisionsKernel(int* grid, int* cellCount, Particle* particles, cui gridWidth, cui gridHeight, cui cellSize, float radius, cui delta)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + delta;
	if (i >= gridHeight * gridWidth) return;
	for (int j{}; j < cellCount[i]; ++j)
	{
		staticGridCollide(i, i, grid, cellCount, particles, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i + 1, grid, cellCount, particles, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i - 1, grid, cellCount, particles, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i + gridHeight, grid, cellCount, particles, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i + gridHeight + 1, grid, cellCount, particles, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i + gridHeight - 1, grid, cellCount, particles, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i - gridHeight, grid, cellCount, particles, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i - gridHeight + 1, grid, cellCount, particles, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i - gridHeight - 1, grid, cellCount, particles, gridWidth, gridHeight, cellSize);
	}
}

__global__ void applyConstraintsKernel(Particle* particles, cui count, sf::Vector2f b1, sf::Vector2f b2)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count) return;
	kernelApplyConstraint(particles[i], b1, b2);
}

__global__ void calculatePositionsKernel(Particle* particles, cui count, float dt)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count) return;
	kernelCalculatePos(particles[i], dt);
}

__global__ void emptyGrid(int* grid, int* cellCount, cui gridWidth, cui gridHeight, cui cellSize)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= gridHeight * gridWidth) return;
	cellCount[i] = 0;
}

KernelScene::KernelScene(cui gw, cui gh, cui cs, const float r, sf::Vector2f b1, sf::Vector2f b2, sf::Vector2f g) :
	gridWidth(gw), gridHeight(gh), cellSize(cs), radius(r), border1(b1), border2(b2), gravity(g)
{

}



void KernelScene::simulate(Particle* p, int count, float dt, int substeps)
{
	Particle* device_particles;
	int* device_grid = 0;
	int* device_cells = 0;

	cudaError_t cudaStatus;

	cudaSetDevice(0);

	cudaStatus = cudaMalloc((void**)&device_grid, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 1 failed %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&device_cells, gridWidth * gridHeight * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 2 failed\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&device_particles, count * sizeof(Particle));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 3 failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMemcpy(device_particles, p, count * sizeof(Particle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy in 3 failed\n");
		goto Error;
	}

	for (int k{}; k < substeps; ++k)
	{
		emptyGrid << < MAX_BLOCKS * NUM_SM, MAX_THREADS_PER_BLOCK >> > (device_grid, device_cells, gridWidth, gridHeight, cellSize);
		cudaDeviceSynchronize();

		applyGravityKernel << < MAX_BLOCKS * NUM_SM, MAX_THREADS_PER_BLOCK >> > (device_particles, count, gravity / (float)substeps);
		cudaDeviceSynchronize();

		for (int n{}; n < 1 + (gridWidth * gridHeight) / MAX_MAX; ++n)
			calculateCollisions << < MAX_BLOCKS * NUM_SM, MAX_THREADS_PER_BLOCK >> >(device_grid, device_cells, device_particles, gridWidth, gridHeight, cellSize, radius, count, n * MAX_MAX);	
		cudaDeviceSynchronize();

		for (int n{}; n < 1 + (gridWidth * gridHeight) / MAX_MAX; ++n)
			correctGrid << < MAX_BLOCKS * NUM_SM, MAX_THREADS_PER_BLOCK >> > (device_cells, gridWidth, gridHeight, cellSize, n);
		cudaDeviceSynchronize();

		for (int n{}; n < 1 + (gridWidth * gridHeight) / MAX_MAX; ++n)
			applyCollisionsKernel << < MAX_BLOCKS * NUM_SM, MAX_THREADS_PER_BLOCK >> > (device_grid, device_cells, device_particles, gridWidth, gridHeight, cellSize, radius, n * MAX_MAX);
		cudaDeviceSynchronize();

		applyConstraintsKernel << < MAX_BLOCKS * NUM_SM, MAX_THREADS_PER_BLOCK >> > (device_particles, count, border1, border2);
		cudaDeviceSynchronize();


		calculatePositionsKernel << < MAX_BLOCKS * NUM_SM, MAX_THREADS_PER_BLOCK >> > (device_particles, count, dt/(float)substeps);
		cudaDeviceSynchronize();
	}


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(p, device_particles, count * sizeof(Particle), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy out failed\n %s", cudaGetErrorString(cudaStatus));
	}


Error:
	cudaFree(device_grid);
	cudaFree(device_cells);
	cudaFree(device_particles);
	
}