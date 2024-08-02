#ifndef GLPA_VECTOR_H_
#define GLPA_VECTOR_H_

#include "Constant.h"

#include <initializer_list>

namespace Glpa
{

class Vector
{
private :
    bool filled = true;

public :
    virtual void toFloatAry(float* ary, int index) = 0;
    virtual void fromFloatAry(float* ary, int index) = 0;

    void empty();
    bool isEmpty();

    void fill();
};

class Vec2d : public Vector
{
private :

public :
    Vec2d(float valueX, float valueY);
    Vec2d(){};

    float x = 0;
    float y = 0;

    void toFloatAry(float* ary, int index) override;
    void fromFloatAry(float* ary, int index) override;

    Vec2d operator+(const Vec2d& other) const 
    {
        return Vec2d(x + other.x, y + other.y);
    }

    Vec2d operator-(const Vec2d& other) const 
    {
        return Vec2d(x - other.x, y - other.y);
    }

    Vec2d operator*(float scalar) const 
    {
        return Vec2d(x * scalar, y * scalar);
    }

    Vec2d operator/(float scalar) const 
    {
        return Vec2d(x / scalar, y / scalar);
    }

    bool operator==(const Vec2d& other) const 
    {
        return x == other.x && y == other.y;
    }

    bool operator!=(const Vec2d& other) const 
    {
        return !(*this == other);
    }

    void operator+=(const Vec2d& other)
    {
        x += other.x;
        y += other.y;
    }

    void operator-=(const Vec2d& other)
    {
        x -= other.x;
        y -= other.y;
    }

    void operator*=(float scalar)
    {
        x *= scalar;
        y *= scalar;
    }

    void operator/=(float scalar)
    {
        x /= scalar;
        y /= scalar;
    }

    Glpa::Vec2d operator=(const std::initializer_list<float>& values)
    {
        if (values.size() == 2) {
            auto it = values.begin();
            x = *it;
            y = *(it + 1);
        }
        return *this;
    }
};

class Vec3d : public Vector
{
private :

public :
    Vec3d(float valueX, float valueY, float valueZ);
    Vec3d(){};

    float x = 0;
    float y = 0;
    float z = 0;

    void toFloatAry(float* ary, int index) override;
    void fromFloatAry(float* ary, int index) override;

};

}

#endif GLPA_VECTOR_H_

