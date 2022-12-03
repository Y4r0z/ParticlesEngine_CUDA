#pragma once
#include <SFML/Graphics.hpp>
#include "../scene/scene.h"

class Renderer
{
private:
	sf::RenderWindow& m_window;
	sf::Texture circleTexture;

	float m_zoom = 1.f;
	sf::View m_view;

public:
	Renderer(sf::RenderWindow &window);
	void render(Scene& scene);

	void listenEvents(sf::Event& event);
};