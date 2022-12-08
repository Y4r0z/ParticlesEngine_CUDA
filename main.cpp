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


int main()
{
    srand((unsigned int)time(0));
    const int W = 1600;
    const int H = 900;
    const int maxObjects = 5000;
    const int radius = 4;

    bool stopped = false;
    bool toDrawPressure = false;
    bool toRender = true;
    
    sf::ContextSettings settings;
    settings.antialiasingLevel = 4;
    sf::RenderWindow window(sf::VideoMode(W, H), "Particles Engine", sf::Style::Default, settings);

    Renderer renderer(window);

    TextManager text(window, 24);


    Scene scene(W, H, radius, sf::Vector2f{ 0, 0 }, sf::Vector2f{ W, H });
    scene.setSubStepsCount(4);


    sf::Clock clock = sf::Clock::Clock();
    sf::Time prev = clock.getElapsedTime();
    sf::Time cur;
    float fps{};



    float ang = 0.f;
    for (int i{}; i < maxObjects; ++i)
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
                if (event.key.code == sf::Keyboard::R)
                    toRender = !toRender;
                if (event.key.code == sf::Keyboard::E)
                    renderer.explode(scene, 80, 30);
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

        cur = clock.getElapsedTime();
        text.drawAll(cur, prev, scene.getCount());
     

        window.display();

        prev = cur;
    }


    return 0;
}
