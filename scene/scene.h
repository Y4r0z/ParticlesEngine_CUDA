#pragma once
#include <vector>
#include "../particle/particle.h"
#include "SFML/Graphics.hpp"
#include "scene.cuh"

class Scene
{
private:
	float m_dt = 0.012f;
	int m_substeps = 4;
	Particle* particles;
	int count = 0;
	sf::Vector2f m_gravity{ 0, 9.8f };
	std::vector<sf::Vector2f> m_borders;

	bool m_pressureMode = false;

	void simpleCollisions();

	//–егул€рна€ сетка
	const int windowWidth;
	const int windowHeight;
	const float radius;

	const int cellSize = 8;
	const int gridWidth = (int)(windowWidth / (radius * 2.f));
	const int gridHeight = (int)(windowHeight / (radius * 2.f));

	KernelScene* kernel;
	

public:
	Scene(int width, int height, float radius, sf::Vector2f b1, sf::Vector2f b2);
	void simulate();

	void addParticle(Particle& p);

	Particle* getParticles();
	std::vector<sf::Vector2f>& getBorders();
	sf::Vector2f getGravity();

	int getCount() { return count; }
	float getRadius() { return radius; }
	int getCellSize() { return cellSize; }

	void setBorders(sf::Vector2f pos1, sf::Vector2f pos2);
	void setGravity(sf::Vector2f g);
	void setSubStepsCount(int cnt) { m_substeps = cnt; }
	void particlesPressure(bool mode);



};