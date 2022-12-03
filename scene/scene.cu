#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "scene.cuh"
#include <iostream>
#include <stdio.h>
#include <SFML/Graphics.hpp>
#include <chrono>
#include <string>

__device__ void pAtomicCpy(Particle& p1, Particle& p2)
{


	atomicExch(&p1.m_radius, p2.m_radius);

	atomicExch(&p1.m_curPos.x, p2.m_curPos.x);
	atomicExch(&p1.m_curPos.y, p2.m_curPos.y);

	atomicExch(&p1.m_prevPos.x, p2.m_prevPos.x);
	atomicExch(&p1.m_prevPos.y, p2.m_prevPos.y);
	
	atomicExch(&p1.m_pressure, p2.m_pressure);
	atomicExch(&p1.m_pressureCoef, p2.m_pressureCoef);

	//atomicExch(&p1.m_color.r, p2.m_color.r);
	//atomicExch(&p1.m_color.g, p2.m_color.g);
	//atomicExch(&p1.m_color.b, p2.m_color.b);

	atomicExch(&p1.m_acceleration.x, p2.m_acceleration.x);
	atomicExch(&p1.m_acceleration.y, p2.m_acceleration.y);

}

__device__ bool pCmp(Particle& p1, Particle& p2)
{
	return p1.m_curPos.x == p2.m_curPos.x && p1.m_curPos.y == p2.m_curPos.y &&
		p1.m_prevPos.x == p2.m_prevPos.x && p1.m_prevPos.y == p2.m_prevPos.y &&
		p1.m_radius == p2.m_radius;
}
__device__ void pCpy(Particle& p1, Particle& other)
{
	p1.m_curPos = other.m_curPos;
	p1.m_prevPos = other.m_prevPos;
	p1.m_radius = other.m_radius;
	p1.m_pressure = other.m_pressure;
	p1.m_pressureCoef = other.m_pressureCoef;
	p1.m_color = other.m_color;
	p1.m_acceleration = other.m_acceleration;
	p1.returnColorPressure = other.returnColorPressure;
}

__device__ void kernelCalculatePos(Particle& p, const float dt)
{
	const float vel_x = p.m_curPos.x - p.m_prevPos.x, vel_y = p.m_curPos.y - p.m_prevPos.y;
	p.m_prevPos.x = p.m_curPos.x;
	p.m_prevPos.y = p.m_curPos.y;
	const float m = sqrtf(vel_x * vel_x + vel_y * vel_y);
	const float mc = p.m_pressure * p.m_pressure;
	const float pc = 1.f / (1.f + mc);

	p.m_curPos.x += (vel_x + (p.m_acceleration.x * dt * dt)) * pc;
	p.m_curPos.y += (vel_y + (p.m_acceleration.y * dt * dt)) * pc;

	p.m_pressureCoef = pc;
	p.m_pressure = 0.f;
}

__device__ void kernelAccelerate(Particle& p, sf::Vector2f a)
{
	p.m_acceleration.x += a.x;
	p.m_acceleration.y += a.y;
}

__device__ void kernelApplyConstraint(Particle& p, sf::Vector2f pos1, sf::Vector2f pos2)
{
	const float
		x = p.m_curPos.x,
		y = p.m_curPos.y,
		r = p.m_radius;
	if (x + r > pos2.x)
		p.m_curPos.x = pos2.x - r;
	if (y + r > pos2.y)
		p.m_curPos.y = pos2.y - r;
	if (x - r < pos1.x)
		p.m_curPos.x = pos1.x + r;
	if (y - r < pos1.y)
		p.m_curPos.y = pos1.y + r;
}

__device__ void kernelCollide(Particle& p1, Particle& p2)
{
	const float ax = p1.m_curPos.x - p2.m_curPos.x;
	const float ay = p1.m_curPos.y - p2.m_curPos.y;
	const float ndist = (ax * ax) + (ay * ay);
	const float r2 = p1.m_radius + p2.m_radius;

	if (ndist > r2 * r2)
		return;
	const float dist = sqrtf(ndist); 
	if (!dist)
		return;
	const float delta = ((r2 - dist)) * 0.5f;
	const float nx = ax / dist * delta;
	const float ny = ay / dist * delta;;
	p1.m_curPos.x += nx;
	p1.m_curPos.y += ny;
	p2.m_curPos.x -= nx;
	p2.m_curPos.y -= ny;
	p1.m_pressure += delta / 2.f;
	p2.m_pressure += delta / 2.f;
	
}



__device__ void staticGridCollide(cui pos1, cui pos2, Particle* grid, int* cellCount, cui gridWidth, cui gridHeight, cui cellSize)
{
	if (pos2 < 0 || pos2 >= gridWidth * gridHeight)
		return;

	for (int i{}; i < cellCount[pos1]; ++i)
		for (int j{}; j < cellCount[pos2]; ++j)
		{
			kernelCollide(grid[pos1 * cellSize + i], grid[pos2 * cellSize + j]);
		}
	
}




__global__ void calculateCollisions(Particle* grid, int* cellCount, cui gridWidth, cui gridHeight, cui cellSize, float radius, int* cellCountOut)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= gridHeight * gridWidth) return;
	for (int j{}; j < cellCount[i]; ++j)
	{
		// Текущая позиция в массиве grid
		const int pos = i * cellSize + j;
		// Реальные координаты
		int x = (int)(grid[pos].m_curPos.x / (radius * 2.f));
		int y = (int)(grid[pos].m_curPos.y / (radius * 2.f));
		// Корреция координат
		if (x < 0)
			x = 0;
		if (y < 0)
			y = 0;
		if (x >= gridWidth)
			x = gridWidth - 1;
		if (y >= gridHeight)
			y = gridHeight - 1;
		// Реальная позиция
		int pos2 = x * gridHeight + y;

		if (pos2 != i && cellCount[pos2] < cellSize)
		{
			// TODO : BUG. Частицы исчезают
			pCpy(grid[pos2 * cellSize + cellCount[pos2]], grid[pos]);
			pCpy(grid[pos], grid[i * cellSize + (cellCount[i] - 1 )]);
			//atomicAdd(&cellCountOut[pos2], 1);
			//atomicSub(&cellCountOut[i], 1);
			cellCount[pos2]++;
			cellCount[i]--;
		}
	}
}


__global__ void applyGravityKernel(Particle* grid, int* cellCount, cui gridWidth, cui gridHeight, cui cellSize, sf::Vector2f g)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= gridHeight * gridWidth) return;
	for (int j{}; j < cellCount[i]; ++j)
		kernelAccelerate(grid[i * cellSize + j], g);
}
__global__ void applyCollisionsKernel(Particle* grid, int* cellCount, cui gridWidth, cui gridHeight, cui cellSize, float radius)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= gridHeight * gridWidth) return;
	for (int j{}; j < cellCount[i]; ++j)
	{
		staticGridCollide(i, i, grid, cellCount, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i + 1, grid, cellCount, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i - 1, grid, cellCount, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i + gridHeight, grid, cellCount, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i + gridHeight + 1, grid, cellCount, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i + gridHeight - 1, grid, cellCount, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i - gridHeight, grid, cellCount, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i - gridHeight + 1, grid, cellCount, gridWidth, gridHeight, cellSize);
		staticGridCollide(i, i - gridHeight - 1, grid, cellCount, gridWidth, gridHeight, cellSize);
	}
}
__global__ void applyConstraintsKernel(Particle* grid, int* cellCount, cui gridWidth, cui gridHeight, cui cellSize, sf::Vector2f b1, sf::Vector2f b2)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= gridHeight * gridWidth) return;
	for (int j{}; j < cellCount[i]; ++j)
		kernelApplyConstraint(grid[i * cellSize + j], b1, b2);
}

__global__ void calculatePositionsKernel(Particle* grid, int* cellCount, cui gridWidth, cui gridHeight, cui cellSize, float dt)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= gridHeight * gridWidth) return;
	for (int j{}; j < cellCount[i]; ++j)
		kernelCalculatePos(grid[i * cellSize + j], dt);
}



KernelScene::KernelScene(cui gw, cui gh, cui cs, const float r, sf::Vector2f b1, sf::Vector2f b2, sf::Vector2f g) : 
	gridWidth(gw), gridHeight(gh), cellSize(cs), radius(r), border1(b1), border2(b2), gravity(g)
{

}



void KernelScene::simulate(Particle* grid, int* cellCount, float dt, int substeps)
{
	sf::Clock clock = sf::Clock::Clock();
	sf::Time prev = clock.getElapsedTime();
	sf::Time cur;

	Particle* device_grid = 0;
	int* device_cells = 0;
	int* device_cells2 = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);

	cudaStatus = cudaMalloc((void**)&device_grid, size * sizeof(Particle));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 1 failed %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&device_cells, gridWidth * gridHeight * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 2 failed\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&device_cells2, gridWidth * gridHeight * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 3 failed\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(device_grid, grid, size * sizeof(Particle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy in 1 failed\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(device_cells, cellCount, gridWidth * gridHeight * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy in 2 failed\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(device_cells2, cellCount, gridWidth * gridHeight * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy in 3 failed\n");
		goto Error;
	}

	//std::cout << "\n1: " << std::to_string(clock.getElapsedTime().asSeconds()) << '\n';
	//kernelSimulate <<< NUM_SM * MAX_BLOCKS, 1024>>> (device_grid, device_cells, gridWidth, gridHeight, cellSize, radius, dt, substeps, border1, border2, gravity);
	for (int k{}; k < 1; ++k)
	{
		applyGravityKernel << < NUM_SM * MAX_BLOCKS, 1024 >> > (device_grid, device_cells, gridWidth, gridHeight, cellSize, gravity);
		cudaDeviceSynchronize();

		calculateCollisions <<< NUM_SM * MAX_BLOCKS, 1024 >>> (device_grid, device_cells, gridWidth, gridHeight, cellSize, radius, device_cells2);
		cudaDeviceSynchronize();	
		applyCollisionsKernel << < NUM_SM * MAX_BLOCKS, 1024 >> > (device_grid, device_cells, gridWidth, gridHeight, cellSize, radius);
		cudaDeviceSynchronize();

		applyConstraintsKernel << < NUM_SM * MAX_BLOCKS, 1024 >> > (device_grid, device_cells, gridWidth, gridHeight, cellSize, border1, border2);
		cudaDeviceSynchronize();

		calculatePositionsKernel << < NUM_SM * MAX_BLOCKS, 1024 >> > (device_grid, device_cells, gridWidth, gridHeight, cellSize, dt/(float)substeps);	
	}
	
	//std::cout << "\n2: " << std::to_string(clock.getElapsedTime().asSeconds()) << '\n';
	//clock = sf::Clock::Clock();
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
	//std::cout << "3: " << std::to_string(clock.getElapsedTime().asSeconds()) << '\n';
	//clock = sf::Clock::Clock();
	cudaStatus = cudaMemcpy(grid, device_grid, size * sizeof(Particle), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy out failed!");
	}
	cudaStatus = cudaMemcpy(cellCount, device_cells, gridWidth * gridHeight * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy out failed!");
	}


	//std::cout << "4: " << std::to_string(clock.getElapsedTime().asSeconds()) << '\n';
Error:
	cudaFree(device_grid);
	cudaFree(device_cells);
	
}