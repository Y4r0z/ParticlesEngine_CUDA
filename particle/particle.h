#pragma once
#include <SFML/Graphics.hpp>
class Particle
{
public:
	float m_radius{1};
	sf::Vector2f m_curPos;
	sf::Vector2f m_prevPos;
	sf::Vector2f m_acceleration;
	sf::Color m_color;
	float m_pressure{0};

	//Debug values
	float m_pressureCoef{0};
	bool returnColorPressure = false;

public:
	Particle();
	Particle(float radius);

	void calculatePos(float dt);
	void accelerate(sf::Vector2f a);
	void applyConstraint(sf::Vector2f pos1, sf::Vector2f pos2);
	void collide(Particle& p);

	sf::Vector2f pos();
	sf::Vector2f prevPos();
	float x();
	float y();
	float r();
	float radius();
	float p();
	float pressure();
	sf::Color color();

	void setPos(sf::Vector2f newPos);
	void setPos(sf::Vector2f newPos, sf::Vector2f prevPos);
	void setRadius(float newRadius);
	void setColor(sf::Color color);
	void setDebugMode(bool mode) { returnColorPressure = mode; }
	


	void print();

    Particle& operator=(const Particle& other)
	{
		if (this == &other) return *this;
		m_curPos = other.m_curPos;
		m_prevPos = other.m_prevPos;
		m_radius = other.m_radius;
		m_pressure = other.m_pressure;
		m_pressureCoef = other.m_pressureCoef;
		m_color = other.m_color;
		m_acceleration = other.m_acceleration;
		returnColorPressure = other.returnColorPressure;
		return *this;
	}


	


};