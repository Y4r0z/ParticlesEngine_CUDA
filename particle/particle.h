#pragma once
#include <SFML/Graphics.hpp>
class Particle
{
public:
	float radius{1};
	sf::Vector2f curPos;
	sf::Vector2f prevPos;
	sf::Vector2f deltaPos;
	sf::Vector2f acceleration;
	sf::Color color;
	float pressure{0};

	//Debug values
	float pressureCoef{0};
	bool returnColorPressure = false;

public:
	Particle();
	Particle(float radius);

	void calculatePos(float dt);
	void accelerate(sf::Vector2f a);
	void applyConstraint(sf::Vector2f pos1, sf::Vector2f pos2);
	void collide(Particle& p);

	sf::Vector2f pos();
	sf::Vector2f getPrevPos();
	float x();
	float y();
	float r();
	float p();
	float getPressure();
	sf::Color getColor();
	void setPos(sf::Vector2f newPos);
	void setPos(sf::Vector2f newPos, sf::Vector2f prevPos);
	void setRadius(float newRadius);
	void setColor(sf::Color color);
	void setDebugMode(bool mode) { returnColorPressure = mode; }
	


	void print();

    Particle& operator=(const Particle& other)
	{
		if (this == &other) return *this;
		curPos = other.curPos;
		prevPos = other.prevPos;
		radius = other.radius;
		pressure = other.pressure;
		pressureCoef = other.pressureCoef;
		color = other.color;
		acceleration = other.acceleration;
		returnColorPressure = other.returnColorPressure;
		deltaPos = other.deltaPos;
		return *this;
	}


	


};