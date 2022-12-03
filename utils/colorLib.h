#pragma once
#include <SFML\Graphics.hpp>
class ColorLib
{
public:
    static float GetHue(float p, float q, float t);
    static sf::Color toRGB(float h, float s, float l); //h[0; 360], s[0; 100], l[0 ;100]

};
