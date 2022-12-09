#include "particle.h"
#include <exception>
#include <iostream>
#include <random>
#include "../utils/colorLib.h";

#define SQRT_MAGIC_F 0x5f3759df 
float sqrtfast(const float x)
{
	const float xhalf = 0.5f * x;

	union // get bits for floating value
	{
		float x;
		int i;
	} u;
	u.x = x;
	u.i = SQRT_MAGIC_F - (u.i >> 1);  // gives initial guess y0
	return x * u.x * (1.5f - xhalf * u.x * u.x);// Newton step, repeating increases accuracy 
}



Particle::Particle(float radius) : radius(radius)
{
	color = ColorLib::toRGB(rand() % 360, 50, 50);
}

Particle::Particle() { color = ColorLib::toRGB(rand() % 360, 50, 50); }

void Particle::calculatePos(float dt)
{
	const sf::Vector2f vel = curPos - prevPos;
	prevPos = curPos;
	//Пытаюсь решить проблемы с давлением
	const float velMagnitude = sqrtf(vel.x * vel.x + vel.y * vel.y);
	const float moveCoef = (pressure * pressure);
	const float pc = 1.f / (1.f + (moveCoef));

	curPos += (vel + (acceleration * (dt * dt))) * pc;

	pressureCoef = pc;
	pressure = 0.f;
}

void Particle::accelerate(sf::Vector2f a) 
{
	acceleration += a;
}

void Particle::applyConstraint(sf::Vector2f pos1, sf::Vector2f pos2)
{
	
	const float 
		x = curPos.x,
		y = curPos.y,
		r = radius;
	if (x + r > pos2.x)
		curPos.x = pos2.x - r;
	if (y + r > pos2.y)
		curPos.y = pos2.y - r;
	if (x - r < pos1.x)
		curPos.x = pos1.x + r;
	if (y - r < pos1.y)
		curPos.y = pos1.y + r;
}

void Particle::collide(Particle& p)
{
	const sf::Vector2f a = curPos - p.curPos;
	const float ndist = (a.x * a.x) + (a.y * a.y);
	const float r2 = radius + p.radius;
	if (ndist > r2 * r2)
		return;
	const float dist = sqrtf(ndist); //Оптиизация вычислений
	//Если что-то не так с ndist, проверка на нуль не сработала
	if (!dist)
		return;
	const float delta = ((r2 - dist)) * 0.5f;
	const sf::Vector2f n{ a.x / dist * delta, a.y / dist * delta };
	curPos += n;
	p.curPos -= n;
	pressure += delta / 2.f;
	p.pressure += delta / 2.f;
	
}




float Particle::x() { return curPos.x; }
float Particle::y() { return curPos.y; }
float Particle::r() { return radius; }
sf::Vector2f Particle::pos() { return curPos; }
float Particle::p() { return pressure; }
float Particle::getPressure() { return pressure; }



void Particle::setPos(sf::Vector2f pos)
{
	curPos = pos;
}
void Particle::setPos(sf::Vector2f pos, sf::Vector2f pPos)
{
	curPos = pos;
	prevPos = pPos;
}

void Particle::setRadius(float r)
{
	radius = r;
}

void Particle::setColor(sf::Color c)
{
	color = c;
}


sf::Color Particle::getColor()
{
	if(returnColorPressure)
		return sf::Color(255, 255 * sqrtf(pressureCoef), 255 * sqrtf(pressureCoef));
	return color;
}
sf::Vector2f Particle::getPrevPos()
{
	return prevPos;
}

void Particle::print() 
{
	const sf::Vector2f vel = (curPos - prevPos);
	std::cout << "Particle."
		<< "Pos: " << curPos.x << ", " << curPos.y << " R: " << radius << 
		"\nPressure coef: " << pressureCoef << 
		"\nVelocity: " << sqrtf(vel.x * vel.x + vel.y * vel.y)
		<< "\n\n";
}