#include <SFML/Graphics.hpp>
#include <iostream>
#include <random>
#include "renderer/renderer.h"
#include "renderer/textManager/textManager.h"
#include <chrono>
#include <string>



/*
hot keys:
E - explode
G - draw debug grid
P - particles pressure map
R - render particles
Space - pause
    Up - simulate 1 step
    PgUp - simulate 1 substep
Mouse1 - get particle info in console
Left/Right - change gravity vector;
A - add more object (+ 1/4 n)
*/
void drawGrid(Scene& scene, sf::RenderWindow& win);
void mouseClick(sf::RenderWindow& win, Scene& scene);
void explode(sf::RenderWindow& win, Scene& scene, float radius, float power);


int main()
{
    srand((unsigned int)time(0));
    const int W = 1600;
    const int H = 900;
    const int maxObjects = 1000;
    const int radius = 4;

    bool stopped = false;
    bool toDrawGrid = false;
    bool toDrawPressure = false;
    bool toRender = true;
    
    sf::ContextSettings settings;
    settings.antialiasingLevel = 4;
    sf::RenderWindow window(sf::VideoMode(W, H), "Particles Engine", sf::Style::Default, settings);

    Renderer renderer(window);

    TextManager text(window, 24);


    Scene scene(W, H, radius, sf::Vector2f{ 0, 0 }, sf::Vector2f{ W, H });
    scene.setSubStepsCount(1);


    sf::Clock clock = sf::Clock::Clock();
    sf::Time prev = clock.getElapsedTime();
    sf::Time cur;
    float fps{};



    float ang = 0.f;
    const int gW = scene.getGridSize().x;
    const int gH = scene.getGridSize().y;
    for (int i{}; i < gW * gH && scene.getParticles().size() < maxObjects; ++i)
    {
        auto p = Particle(radius);
        auto pos = sf::Vector2f(rand() % W, rand() % H);
        p.setPos(pos, pos);
        scene.addParticle(p);
    }

    while (window.isOpen())
    {
        sf::Event event{};
        while (window.pollEvent(event))
        {
            renderer.listenEvents(event);

            if (event.type == sf::Event::Closed)
                window.close();
            if (event.type == sf::Event::KeyPressed)
            {
                if(event.key.code == sf::Keyboard::Space)
                    stopped = !stopped;
                if (event.key.code == sf::Keyboard::Up && stopped)
                    scene.simulate();
                //if (event.key.code == sf::Keyboard::PageUp && stopped)
                    //scene.subSimulate(); Пока не реализовано на GPU
                if (event.key.code == sf::Keyboard::G)
                    toDrawGrid = !toDrawGrid;
                if (event.key.code == sf::Keyboard::R)
                    toRender = !toRender;
                if (event.key.code == sf::Keyboard::E)
                    explode(window, scene, 120, 200);
                if (event.key.code == sf::Keyboard::P)
                    toDrawPressure = !toDrawPressure;
                    scene.particlesPressure(toDrawPressure);
                if(event.key.code == sf::Keyboard::A)
                {
                    for (int i = 0; i < maxObjects / 4; ++i)
                    {
                        auto p = Particle(radius);
                        auto pos = sf::Vector2f(rand() % W, rand() % H);
                        p.setPos(pos, pos);
                        scene.addParticle(p);
                    }
                }
            }
        }
         
        window.clear();
        if(!stopped)
            scene.simulate();
        if(toRender)
            renderer.render(scene);
        if (toDrawGrid)
            drawGrid(scene, window);


        cur = clock.getElapsedTime();
        text.drawAll(cur, prev, scene.getParticles().size());
     

        window.display();

        prev = cur;
    }


    return 0;
}


void drawGrid(Scene& scene, sf::RenderWindow& win)
{
    int  X = scene.getGridSize().x;
    int Y = scene.getGridSize().y;

    auto W = scene.getBorders()[1].x;
    auto H = scene.getBorders()[1].y;

    auto sW = W / X;
    auto sH = H / Y;

    auto cellCnt = scene.getGridCellCount();

    sf::RectangleShape r(sf::Vector2f(sW, sH));
    r.setFillColor(sf::Color::Black);
    r.setOutlineThickness(0.5f);
    r.setOutlineColor(sf::Color(255,255,255,100));

    const int size = X * Y;

    for (int i{}; i < size; ++i)
    {
        float color = cellCnt[i] * 100;
        float alpha = cellCnt[i] * 90;
        if (color > 255) color = 255;
        if (alpha > 255) alpha = 255;
        r.setFillColor(sf::Color(0, color, 0, alpha));
        r.setPosition(sf::Vector2f((i/Y) * sW, (i%Y) * sH));
        win.draw(r);
    }   
}

void mouseClick(sf::RenderWindow& win, Scene& scene)
{
    const auto pos = sf::Vector2f(sf::Mouse::getPosition(win).x, sf::Mouse::getPosition(win).y);
    std::cout << pos.x << ' ' << pos.y << '\n';
    for (auto i : scene.getParticles())
    {
        const sf::Vector2f diff = pos - i.pos();
        if (sqrtf(diff.x * diff.x + diff.y * diff.y) <= i.radius())
        {
            i.print();
        }
    }
}

void explode(sf::RenderWindow& win, Scene& scene, float radius, float power)
{
    const auto pos = sf::Vector2f(sf::Mouse::getPosition(win).x, sf::Mouse::getPosition(win).y);
    std::cout << pos.x << ' ' << pos.y << '\n' << "Explosion" << "\n";

    auto gridWidth = scene.getGridSize().x;
    auto gridHeight = scene.getGridSize().y;
    auto cellCount = scene.getGridCellCount();
    auto grid = scene.getGrid();
    auto cellSize = scene.getCellSize();
    int cnt = 0;
    for (int i{}; i < gridWidth * gridHeight; ++i)
        for (int j{}; j < cellCount[i]; ++j)
        {
            cnt++;
            const int index = i * cellSize + j;
            const sf::Vector2f diff = pos - grid[index].pos();
            const float m = sqrtf(diff.x * diff.x + diff.y * diff.y);
            if (m <= radius)
            {
                grid[index].setPos(grid[index].pos(), grid[index].pos() + (diff / m * (radius - (m * 0.7f)) / radius * power));
            }

        }
    std::cout << cnt << " : " << scene.getParticles().size() << '\n';

}