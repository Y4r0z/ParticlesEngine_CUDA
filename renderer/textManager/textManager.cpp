#include "textManager.h"

TextManager::TextManager(sf::RenderWindow& window, int size): m_window(window)
{
    auto color = sf::Color::Red;

	font.loadFromFile("resources/Roboto.ttf");

    m_fps.setFont(font);
    m_fps.setCharacterSize(size);
    m_fps.setFillColor(color);
    m_fps.setPosition(0, 0);

    m_frameTime.setFont(font);
    m_frameTime.setCharacterSize(size);
    m_frameTime.setFillColor(color);
    m_frameTime.setPosition(0, size + 1);

    m_particles.setFont(font);
    m_particles.setCharacterSize(size);
    m_particles.setFillColor(color);
    m_particles.setPosition(0, size * 2 + 1);
}

void TextManager::drawFps(sf::Time& cur, sf::Time& prev)
{
    auto fps = 1.f / (cur.asSeconds() - prev.asSeconds());
    m_fps.setString("FPS: " + std::to_string((int)fps));
    m_window.draw(m_fps);
}

void TextManager::drawFrameTime(sf::Time& cur, sf::Time& prev)
{
    m_frameTime.setString("Frame time: " + std::to_string((int)((cur.asSeconds() - prev.asSeconds()) * 1000)));
    m_window.draw(m_frameTime);
}

void TextManager::drawParticlesCount(int cnt)
{
    m_particles.setString("Particles: " + std::to_string(cnt));
    m_window.draw(m_particles);
}

void TextManager::drawAll(sf::Time& cur, sf::Time& prev, int particlesCount)
{
    drawFps(cur, prev);
    drawFrameTime(cur, prev);
    drawParticlesCount(particlesCount);
}