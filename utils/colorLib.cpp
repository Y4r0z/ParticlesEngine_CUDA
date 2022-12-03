#include "colorLib.h";
#include <iostream>;

float ColorLib::GetHue(float p, float q, float t)
{
    float value = p;

    if (t < 0.f) t++;
    if (t > 1.f) t--;

    if (t < 1.f / 6.f)
    {
        value = p + (q - p) * 6 * t;
    }
    else if (t < 1.f / 2.f)
    {
        value = q;
    }
    else if (t < 2.f / 3.f)
    {
        value = p + (q - p) * (2.f / 3.f - t) * 6.f;
    }

    return value;
}

sf::Color  ColorLib::toRGB(float h, float s, float l) //h[0; 360], s[0; 100], l[0 ;100]
{
    float mH, mS, mL, r = 1, g = 1, b = 1, q, p;

    mH = h / 360.f;
    mS = s / 100.f;
    mL = l / 100.f;

    q = (mL < 0.5) ? mL * (1 + mS) : mL + mS - mL * mS;
    p = 2 * mL - q;

    if (mL == 0)
    {
        r = 0;
        g = 0;
        b = 0;
    }
    else if (mS != 0)
    {
        r = GetHue(p, q, mH + 1.f / 3.f);
        g = GetHue(p, q, mH);
        b = GetHue(p, q, mH - 1.f / 3.f);
    }
    sf::Color col(r * 255.0f, g * 255.0f, b * 255.0f);
    return col;
}