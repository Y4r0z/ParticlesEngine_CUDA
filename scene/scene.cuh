#pragma once
#include "../particle/particle.h"

using cui = const unsigned int;
constexpr auto MAX_THREADS_PER_BLOCK = 1024;
constexpr auto MAX_BLOCKS = 16;
constexpr auto NUM_SM = 5;
constexpr auto MAX_MAX = MAX_THREADS_PER_BLOCK * MAX_BLOCKS * NUM_SM;
class KernelScene
{
private:
	cui gridWidth;
	cui gridHeight;
	cui cellSize;
	const float radius;
	cui size = gridWidth * gridHeight * cellSize;
	sf::Vector2f gravity;
	sf::Vector2f border1;
	sf::Vector2f border2;



public:
	KernelScene(cui gw, cui gh, cui cs, const float r, sf::Vector2f border1, sf::Vector2f border2, sf::Vector2f gravity);
	void simulate(Particle* p, int count, float dt, int substeps);
};


