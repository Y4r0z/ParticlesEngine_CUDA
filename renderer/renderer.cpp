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
	auto particles = scene.getParticles();
	int count = scene.getCount();
	sf::VertexArray v(sf::Quads, count * 4);

	for (size_t i{}; i < count * 4; i += 4)
	{
		const auto pos = particles[i / 4].pos();
		const auto r = particles[i / 4].r();
		const auto c = particles[i / 4].getColor();
		const int t = 64; //Размер текстуры

		v[i].color = c;
		v[i].position = sf::Vector2f(pos.x - r, pos.y - r);
		v[i].texCoords = sf::Vector2f(0, 0);

		v[i + 1].color = c;
		v[i + 1].position = sf::Vector2f(pos.x + r, pos.y - r);
		v[i + 1].texCoords = sf::Vector2f(t, 0);

		v[i + 2].color = c;
		v[i + 2].position = sf::Vector2f(pos.x + r, pos.y + r);
		v[i + 2].texCoords = sf::Vector2f(t, t);

		v[i + 3].color = c;
		v[i + 3].position = sf::Vector2f(pos.x - r, pos.y + r);
		v[i + 3].texCoords = sf::Vector2f(0, t);

	}
	m_window.draw(v, &circleTexture);

}

void Renderer::listenEvents(sf::Event& event)
{
	if (event.type == sf::Event::MouseWheelMoved)
	{
		auto delta = event.mouseWheel.delta;
		m_zoom = 1 - (float)delta/8.f;
		m_view.zoom(m_zoom);
		m_window.setView(m_view);
	}
	if (event.type == sf::Event::MouseButtonPressed)
	{
		moving = true;
		mouseFixedPos = m_window.mapPixelToCoords(sf::Vector2i(event.mouseButton.x, event.mouseButton.y));
	}
	else if (event.type == sf::Event::MouseButtonReleased)
	{
		moving = false;
	}
	else if (event.type == sf::Event::MouseMoved)
	{
		if (!moving) return;
		const sf::Vector2f newPos = m_window.mapPixelToCoords(sf::Vector2i(event.mouseMove.x, event.mouseMove.y));
		const sf::Vector2f deltaPos = mouseFixedPos - newPos;

		m_view.setCenter(m_view.getCenter() + deltaPos);
		m_window.setView(m_view);

		mouseFixedPos = m_window.mapPixelToCoords(sf::Vector2i(event.mouseMove.x, event.mouseMove.y));
	}
}	

void Renderer::explode(Scene& scene, float radius, float power)
{
	const auto pos = sf::Vector2f(sf::Mouse::getPosition(m_window).x, sf::Mouse::getPosition(m_window).y);
	//std::cout << pos.x << ' ' << pos.y << ": " << "Explosion" << "\n";
	auto particles = scene.getParticles();
	int count = scene.getCount();
	for (int i{}; i < count; ++i)
	{
		sf::Vector2f diff = pos - particles[i].pos() - (sf::Vector2f)m_window.getSize()/2.f + m_view.getCenter();
		const float m = sqrtf(diff.x * diff.x + diff.y * diff.y);
		if (m <= radius)
		{
			particles[i].setPos(particles[i].pos(), particles[i].pos() + (diff / m * (radius - (m * 0.7f)) / radius * power));
		}
	}

}