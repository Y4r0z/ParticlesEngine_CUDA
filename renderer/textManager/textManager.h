#pragma once
#include <SFML/Graphics.hpp>

class TextManager
{
private:
	sf::Font font;
	sf::RenderWindow& m_window;

	sf::Text m_fps;
	sf::Text m_frameTime;
	sf::Text m_particles;
public:
	TextManager(sf::RenderWindow& window, int size);

	void drawFps(sf::Time& cur, sf::Time& prev);
	void drawFrameTime(sf::Time& cur, sf::Time& prev);
	void drawParticlesCount(int cnt);

	void drawAll(sf::Time& cur, sf::Time& prev, int particlesCount);
};