#include "Vector.cuh"

Glpa::Vec2d::Vec2d(float valueX, float valueY)
{
    x = valueX;
    y = valueY;
}

void Glpa::Vec2d::toFloatAry(float *ary, int index)
{
    ary[index * Glpa::SIZE_VEC2D + Glpa::X] = x;
    ary[index * Glpa::SIZE_VEC2D + Glpa::Y] = y;
}

void Glpa::Vec2d::fromFloatAry(float *ary, int index)
{
    x = ary[index * Glpa::SIZE_VEC2D + Glpa::X];
    y = ary[index * Glpa::SIZE_VEC2D + Glpa::Y];
}

Glpa::Vec3d::Vec3d(float valueX, float valueY, float valueZ)
{
    x = valueX;
    y = valueY;
    z = valueZ;
}

void Glpa::Vec3d::toFloatAry(float *ary, int index)
{
    ary[index * Glpa::SIZE_VEC3D + Glpa::X] = x;
    ary[index * Glpa::SIZE_VEC3D + Glpa::Y] = y;
    ary[index * Glpa::SIZE_VEC3D + Glpa::Z] = z;
}

void Glpa::Vec3d::fromFloatAry(float *ary, int index)
{
    x = ary[index * Glpa::SIZE_VEC3D + Glpa::X];
    y = ary[index * Glpa::SIZE_VEC3D + Glpa::Y];
    z = ary[index * Glpa::SIZE_VEC3D + Glpa::Z];
}

void Glpa::Vector::empty()
{
    filled = false;
}

bool Glpa::Vector::isEmpty()
{
    if (filled) return false;
    else return true;
}

void Glpa::Vector::fill()
{
    filled = true;
}
