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

	bool moving = false;
	sf::Vector2f mouseFixedPos{};

public:
	Renderer(sf::RenderWindow &window);
	void render(Scene& scene);
	void explode(Scene& scene, float radius, float power);
	void listenEvents(sf::Event& event);
};