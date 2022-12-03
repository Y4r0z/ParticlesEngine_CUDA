
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "scene.h"
#include <iostream>

#include "scene.cuh"


Scene::Scene(int width, int height, float radius, sf::Vector2f b1, sf::Vector2f b2) :  windowWidth(width), windowHeight(height), radius(radius)
{
	m_borders.push_back(b1); m_borders.push_back(b2);
	grid = new Particle [gridWidth * gridHeight * cellSize];
	cellCount = new int[gridWidth * gridHeight];
	for (int i{}; i < gridWidth * gridHeight; ++i)
		cellCount[i] = 0;
	kernel = new KernelScene(gridWidth, gridHeight, cellSize, radius, m_borders[0], m_borders[1], m_gravity);
}

std::vector<Particle>& Scene::getParticles()
{
	return m_particles;
}

std::vector<sf::Vector2f>& Scene::getBorders()
{
	return m_borders;
}


void Scene::setBorders(sf::Vector2f pos1, sf::Vector2f pos2)
{
	m_borders.clear();
	m_borders.push_back(pos1);
	m_borders.push_back(pos2);
}

void Scene::setGravity(sf::Vector2f g)
{
	m_gravity = g;
}


void Scene::addParticle(Particle& p)
{
	p.setDebugMode(m_pressureMode);
	if (radius > 0.f)
		p.setRadius(radius);
	int x = (int)(p.x() / (radius * 2.f));
	int y = (int)(p.y() / (radius * 2.f));
	if (x < 0)
		x = 0;
	if (y < 0)
		y = 0;
	if (x >= gridWidth)
		x = gridWidth - 1;
	if (y >= gridHeight)
		y = gridHeight - 1;
	const int pos = x * gridHeight + y;
	if (cellCount[pos] < cellSize)
	{
		grid[pos * cellSize + cellCount[pos]++] = p;
		m_particles.push_back(p);
	}
}





void Scene::simulate()
{
	kernel->simulate(grid, cellCount, m_dt, m_substeps);
}

void Scene::subSimulate()
{
	float sub_dt = m_dt / (float)m_substeps;
	applyGravity();
	applyCollisions();
	applyConstraints();
	updatePositions(sub_dt);
}

void Scene::applyGravity()
{

	for (int i{}; i < gridWidth * gridHeight; ++i)
		for (int j{}; j < cellCount[i]; ++j)
		{
			const int pos = i * cellSize + j;
			grid[pos].accelerate(m_gravity);
		}
}

void Scene::updatePositions(float dt)
{
	for (int i{}; i < gridWidth * gridHeight; ++i)
		for (int j{}; j < cellCount[i]; ++j)
		{
			const int pos = i * cellSize + j;
			grid[pos].calculatePos(dt);
		}

}

void Scene::applyConstraints()
{
	
	for (int i{}; i < gridWidth * gridHeight; ++i)
		for (int j{}; j < cellCount[i]; ++j)
		{
			const int pos = i * cellSize + j;
			grid[pos].applyConstraint(m_borders[0], m_borders[1]);
		}
}

void Scene::simpleCollisions()
{
	for (int i{}; i < m_particles.size(); ++i)
		for (int j{ i + 1 }; j < m_particles.size(); ++j)
			m_particles[i].collide(m_particles[j]);
}




void Scene::gridCollide(int pos1, int pos2)
{
	const int size = gridWidth * gridHeight;
	if (pos2 < 0 || pos2 >= size)
		return;

	for(int i{}; i < cellCount[pos1]; ++i)
		for (int j{}; j < cellCount[pos2]; ++j)
		{
			grid[pos1 * cellSize + i].collide(grid[pos2 * cellSize + j]);
		}
}



void Scene::updateGrid()
{
	//Полное обновление сетки
	for (int i{}; i < gridWidth * gridHeight; ++i)
		cellCount[i] = 0;

	for (auto& i : m_particles)
	{
		// Позиция частицы в сетке
		int x = (int)(i.x() / (radius * 2.f));
		int y = (int)(i.y() / (radius * 2.f));
		//Ограничение позиции, чтобы не выйти за пределы массива
		if (x < 0)
			x = 0;
		if (y < 0)
			y = 0;
		if (x >= gridWidth)
			x = gridWidth - 1;
		if (y >= gridHeight)
			y = gridHeight - 1;
		// Позиция в массиве
		const int pos = x * gridHeight + y;
		// Запись в сетку
		if (cellCount[pos] < cellSize)
		{
			grid[pos * cellSize + cellCount[pos]++] = i;
		}
	}
}

void Scene::staticGrid()
{
	//kernel->cudaGridCollide(grid, cellCount);
}


void Scene::applyCollisions()
{
	//simpleCollisions();
	staticGrid();
	
}

sf::Vector2f Scene::getGravity()
{
	return m_gravity;
}

sf::Vector2f Scene::getGridSize()
{
	return sf::Vector2f(gridWidth, gridHeight);
}

Particle* Scene::getGrid()
{
	return grid;
}

int* Scene::getGridCellCount()
{
	return cellCount;
}

void Scene::particlesPressure(bool mode)
{
	m_pressureMode = mode;
	for (int i{}; i < gridWidth * gridHeight; ++i)
		for (int j{}; j < cellCount[i]; ++j)
			grid[i * cellSize + j].returnColorPressure = mode;
		
}
