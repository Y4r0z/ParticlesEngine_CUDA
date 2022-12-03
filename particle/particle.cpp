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



Particle::Particle(float radius) : m_radius(radius)
{
	m_color = ColorLib::toRGB(rand() % 360, 50, 50);
}

Particle::Particle() { m_color = ColorLib::toRGB(rand() % 360, 50, 50); }

void Particle::calculatePos(float dt)
{
	const sf::Vector2f vel = m_curPos - m_prevPos;
	m_prevPos = m_curPos;
	//Пытаюсь решить проблемы с давлением
	const float velMagnitude = sqrtf(vel.x * vel.x + vel.y * vel.y);
	const float moveCoef = (m_pressure * m_pressure);
	const float pressureCoef = 1.f / (1.f + (moveCoef));

	m_curPos += (vel + (m_acceleration * (dt * dt))) * pressureCoef;

	m_pressureCoef = pressureCoef;
	m_pressure = 0.f;
}

void Particle::accelerate(sf::Vector2f a) 
{
	m_acceleration += a;
}

void Particle::applyConstraint(sf::Vector2f pos1, sf::Vector2f pos2)
{
	
	const float 
		x = m_curPos.x,
		y = m_curPos.y,
		r = m_radius;
	if (x + r > pos2.x)
		m_curPos.x = pos2.x - r;
	if (y + r > pos2.y)
		m_curPos.y = pos2.y - r;
	if (x - r < pos1.x)
		m_curPos.x = pos1.x + r;
	if (y - r < pos1.y)
		m_curPos.y = pos1.y + r;
}

void Particle::collide(Particle& p)
{
	const sf::Vector2f a = m_curPos - p.m_curPos;
	const float ndist = (a.x * a.x) + (a.y * a.y);
	const float r2 = m_radius + p.m_radius;
	if (ndist > r2 * r2)
		return;
	const float dist = sqrtf(ndist); //Оптиизация вычислений
	//Если что-то не так с ndist, проверка на нуль не сработала
	if (!dist)
		return;
	const float delta = ((r2 - dist)) * 0.5f;
	const sf::Vector2f n{ a.x / dist * delta, a.y / dist * delta };
	m_curPos += n;
	p.m_curPos -= n;
	m_pressure += delta / 2.f;
	p.m_pressure += delta / 2.f;
	
}




float Particle::x() { return m_curPos.x; }
float Particle::y() { return m_curPos.y; }
float Particle::r() { return m_radius; }
float Particle::radius() { return m_radius; }
sf::Vector2f Particle::pos() { return m_curPos; }
float Particle::p() { return m_pressure; }
float Particle::pressure() { return m_pressure; }



void Particle::setPos(sf::Vector2f curPos)
{
	m_curPos = curPos;
}
void Particle::setPos(sf::Vector2f curPos, sf::Vector2f prevPos)
{
	m_curPos = curPos;
	m_prevPos = prevPos;
}

void Particle::setRadius(float radius)
{
	m_radius = radius;
}

void Particle::setColor(sf::Color color)
{
	m_color = color;
}


sf::Color Particle::color()
{
	if(returnColorPressure)
		return sf::Color(255, 255 * sqrtf(m_pressureCoef), 255 * sqrtf(m_pressureCoef));
	return m_color;
}
sf::Vector2f Particle::prevPos()
{
	return m_prevPos;
}

void Particle::print() 
{
	const sf::Vector2f vel = (m_curPos - m_prevPos);
	std::cout << "Particle."
		<< "Pos: " << m_curPos.x << ", " << m_curPos.y << " R: " << m_radius << 
		"\nPressure coef: " << m_pressureCoef << 
		"\nVelocity: " << sqrtf(vel.x * vel.x + vel.y * vel.y)
		<< "\n\n";
}