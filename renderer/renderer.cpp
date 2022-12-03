#include "renderer.h"
#include <iostream>
#include <exception>
#include <vector>


Renderer::Renderer(sf::RenderWindow& window) : m_window(window)
{
	m_view = sf::View(sf::Vector2f(window.getSize().x / 2, window.getSize().y / 2), sf::Vector2f(1600.f, 900.f));
	window.setView(m_view);
	window.setFramerateLimit(60);


	if (!circleTexture.loadFromFile("resources/circle.png"))
	{
		throw std::exception("Can't open the circle texture!");
	}
}



void Renderer::render(Scene& scene)
{
	auto gridWidth = scene.getGridSize().x;
	auto gridHeight = scene.getGridSize().y;
	auto cellCount = scene.getGridCellCount();
	auto grid = scene.getGrid();
	auto cellSize = scene.getCellSize();

	const int t = 64; //tex size
	sf::VertexArray va(sf::Quads);

	for (int i{}; i < gridWidth * gridHeight; ++i)
		for (int j{}; j < cellCount[i]; ++j)
		{
			const int index = i * cellSize + j;
			const auto pos = grid[index].pos();
			const auto r = grid[index].r();
			const auto c = grid[index].color();

			sf::Vertex v1, v2, v3, v4;

			v1.color = c;
			v1.position = sf::Vector2f(pos.x - r, pos.y - r);
			v1.texCoords = sf::Vector2f(0, 0);

			v2.color = c;
			v2.position = sf::Vector2f(pos.x + r, pos.y - r);
			v2.texCoords = sf::Vector2f(t, 0);

			v3.color = c;
			v3.position = sf::Vector2f(pos.x + r, pos.y + r);
			v3.texCoords = sf::Vector2f(t, t);

			v4.color = c;
			v4.position = sf::Vector2f(pos.x - r, pos.y + r);
			v4.texCoords = sf::Vector2f(0, t);

			va.append(v1); va.append(v2); va.append(v3); va.append(v4);
			
		}

	m_window.draw(va, &circleTexture);
	
	/*
	sf::CircleShape c{ 1.f };
	c.setPointCount(16);
	c.setOrigin(1.f, 1.f);
	const auto& particles = scene.getParticles();
	for (const auto& i : particles)
	{
		c.setPosition(i.pos().x, i.pos().y);
		c.setScale(i.radius(), i.radius());
		c.setFillColor(i.color());
		m_window.draw(c);


	}
	*/
	
}

void Renderer::listenEvents(sf::Event& event)
{
	if (event.type == sf::Event::MouseWheelMoved)
	{
		auto delta = event.mouseWheel.delta;
		m_zoom -= (float)delta / 40.f;
		sf::View view = m_window.getDefaultView();
		view.zoom(m_zoom);
		m_window.setView(view);
	}
	if (event.type == sf::Event::MouseButtonPressed)
	{

	}
}