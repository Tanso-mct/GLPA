#ifndef GLPA_VECTOR_H_
#define GLPA_VECTOR_H_

#include "Constant.h"

namespace Glpa
{

class Vector
{
private :

public :
    virtual void toFloatAry(float* ary, int index) = 0;
    virtual void fromFloatAry(float* ary, int index) = 0;

};

class Vec2d : public Vector
{
private :

public :
    Vec2d(float valueX, float valueY);

    float x = 0;
    float y = 0;

    void toFloatAry(float* ary, int index) override;
    void fromFloatAry(float* ary, int index) override;

};

class Vec3d : public Vector
{
private :

public :
    Vec3d(float valueX, float valueY, float valueZ);

    float x = 0;
    float y = 0;
    float z = 0;

    void toFloatAry(float* ary, int index) override;
    void fromFloatAry(float* ary, int index) override;

};

}

#endif GLPA_VECTOR_H_

