
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "scene.h"
#include <iostream>

#include "scene.cuh"


Scene::Scene(int width, int height, float radius, sf::Vector2f b1, sf::Vector2f b2) :  windowWidth(width), windowHeight(height), radius(radius)
{
	m_borders.push_back(b1); m_borders.push_back(b2);
	particles = new Particle [gridWidth * gridHeight * cellSize];
	kernel = new KernelScene(gridWidth, gridHeight, cellSize, radius, m_borders[0], m_borders[1], m_gravity);
}

Particle* Scene::getParticles()
{
	return particles;
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
	particles[count++] = p;
}





void Scene::simulate()
{
	kernel->simulate(particles, count, m_dt, m_substeps);
}

sf::Vector2f Scene::getGravity()
{
	return m_gravity;
}


void Scene::particlesPressure(bool mode)
{
	m_pressureMode = mode;
	for (int i{}; i < count; ++i)
		particles[i].returnColorPressure = mode;	
}
