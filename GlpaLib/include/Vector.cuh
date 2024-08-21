#ifndef GLPA_VECTOR_H_
#define GLPA_VECTOR_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Constant.h"
#include <cmath>

#include <initializer_list>
#include <Windows.h>

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

typedef struct _GPU_VEC_2D
{
    float x;
    float y;

    __device__ __host__ _GPU_VEC_2D() : x(0), y(0) {}
    __device__ __host__ _GPU_VEC_2D(float x, float y) : x(x), y(y) {}

    __device__ __host__ _GPU_VEC_2D operator+(const _GPU_VEC_2D& other) const
    {
        return _GPU_VEC_2D(x + other.x, y + other.y);
    }

    __device__ __host__ _GPU_VEC_2D operator-(const _GPU_VEC_2D& other) const
    {
        return _GPU_VEC_2D(x - other.x, y - other.y);
    }

    __device__ __host__ _GPU_VEC_2D operator*(float scalar) const
    {
        return _GPU_VEC_2D(x * scalar, y * scalar);
    }

    __device__ __host__ _GPU_VEC_2D operator/(float scalar) const
    {
        return _GPU_VEC_2D(x / scalar, y / scalar);
    }

    __device__ __host__ GPU_BOOL operator==(const _GPU_VEC_2D& other) const
    {
        return (x == other.x && y == other.y) ? TRUE : FALSE;
    }

    __device__ __host__ GPU_BOOL operator!=(const _GPU_VEC_2D& other) const
    {
        return (!(*this == other)) ? TRUE : FALSE;
    }

    __device__ __host__ void  operator+=(const _GPU_VEC_2D& other)
    {
        x += other.x;
        y += other.y;
    }

    __device__ __host__ void  operator-=(const _GPU_VEC_2D& other)
    {
        x -= other.x;
        y -= other.y;
    }

    __device__ __host__ void  operator*=(float scalar)
    {
        x *= scalar;
        y *= scalar;
    }

    __device__ __host__ void  operator/=(float scalar)
    {
        x /= scalar;
        y /= scalar;
    }

    __device__ __host__ _GPU_VEC_2D operator=(const _GPU_VEC_2D& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }
} GPU_VEC_2D;

class Vec2d : public Vector
{
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

typedef struct _GPU_VEC_3D
{
    float x;
    float y;
    float z;

    __device__ __host__ _GPU_VEC_3D() : x(0), y(0), z(0) {}
    __device__ __host__ _GPU_VEC_3D(float x, float y, float z) : x(x), y(y), z(z) {}

    __device__ __host__ _GPU_VEC_3D operator+(const _GPU_VEC_3D& other) const
    {
        return _GPU_VEC_3D(x + other.x, y + other.y, z + other.z);
    }

    __device__ __host__ _GPU_VEC_3D operator-(const _GPU_VEC_3D& other) const
    {
        return _GPU_VEC_3D(x - other.x, y - other.y, z - other.z);
    }

    __device__ __host__ _GPU_VEC_3D operator*(float scalar) const
    {
        return _GPU_VEC_3D(x * scalar, y * scalar, z * scalar);
    }

    __device__ __host__ _GPU_VEC_3D operator/(float scalar) const
    {
        return _GPU_VEC_3D(x / scalar, y / scalar, z / scalar);
    }

    __device__ __host__ GPU_BOOL operator==(const _GPU_VEC_3D& other) const
    {
        return (x == other.x && y == other.y && z == other.z) ? TRUE : FALSE;
    }

    __device__ __host__ GPU_BOOL operator!=(const _GPU_VEC_3D& other) const
    {
        return (!(*this == other)) ? TRUE : FALSE;
    }

    __device__ __host__ void  operator+=(const _GPU_VEC_3D& other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
    }

    __device__ __host__ void  operator-=(const _GPU_VEC_3D& other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;
    }

    __device__ __host__ void  operator*=(float scalar)
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
    }

    __device__ __host__ void  operator/=(float scalar)
    {
        x /= scalar;
        y /= scalar;
        z /= scalar;
    }

    __device__ __host__ _GPU_VEC_3D operator=(const _GPU_VEC_3D& other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }
    
} GPU_VEC_3D;

class Vec3d : public Vector
{
public :
    Vec3d(float valueX, float valueY, float valueZ);
    Vec3d(){};

    float x = 0;
    float y = 0;
    float z = 0;

    void toFloatAry(float* ary, int index) override;
    void fromFloatAry(float* ary, int index) override;

    Vec3d operator+(const Vec3d& other) const 
    {
        return Vec3d(x + other.x, y + other.y, z + other.z);
    }

    Vec3d operator-(const Vec3d& other) const 
    {
        return Vec3d(x - other.x, y - other.y, z - other.z);
    }

    Vec3d operator*(float scalar) const 
    {
        return Vec3d(x * scalar, y * scalar, z * scalar);
    }

    Vec3d operator/(float scalar) const 
    {
        return Vec3d(x / scalar, y / scalar, z / scalar);
    }

    bool operator==(const Vec3d& other) const 
    {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator!=(const Vec3d& other) const 
    {
        return !(*this == other);
    }

    void operator+=(const Vec3d& other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
    }

    void operator-=(const Vec3d& other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;
    }

    void operator*=(float scalar)
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
    }

    void operator/=(float scalar)
    {
        x /= scalar;
        y /= scalar;
        z /= scalar;
    }

    Glpa::Vec3d operator=(const std::initializer_list<float>& values)
    {
        if (values.size() == 2) {
            auto it = values.begin();
            x = *it;
            y = *(it + 1);
            z = *(it + 2);
        }
        return *this;
    }
};

typedef struct _GPU_VECTOR_MANAGER
{
    // Vector 2D
    __device__ __host__ float dot(Glpa::GPU_VEC_2D vec1, Glpa::GPU_VEC_2D vec2)
    {
        return vec1.x * vec2.x + vec1.y * vec2.y;
    }

    __device__ __host__ float cross(Glpa::GPU_VEC_2D vec1, Glpa::GPU_VEC_2D vec2)
    {
        return vec1.x * vec2.y - vec1.y * vec2.x;
    }

    __device__ __host__ float getLength(Glpa::GPU_VEC_2D vec)
    {
        return std::sqrtf(vec.x * vec.x + vec.y * vec.y);
    }

    __device__ __host__ Glpa::GPU_VEC_2D normalize(Glpa::GPU_VEC_2D vec)
    {
        float length = getLength(vec);
        return Glpa::GPU_VEC_2D(vec.x / length, vec.y / length);
    }

    __device__ __host__ float cos(Glpa::GPU_VEC_2D vec1, Glpa::GPU_VEC_2D vec2)
    {
        return dot(vec1, vec2) / (getLength(vec1) * getLength(vec2));
    }

    // Vector 3D
    __device__ __host__ float dot(Glpa::GPU_VEC_3D vec1, Glpa::GPU_VEC_3D vec2)
    {
        return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
    }

    __device__ __host__ Glpa::GPU_VEC_3D cross(Glpa::GPU_VEC_3D vec1, Glpa::GPU_VEC_3D vec2)
    {
        return Glpa::GPU_VEC_3D(vec1.y * vec2.z - vec1.z * vec2.y,
                                vec1.z * vec2.x - vec1.x * vec2.z,
                                vec1.x * vec2.y - vec1.y * vec2.x);
    }

    __device__ __host__ float getLength(Glpa::GPU_VEC_3D vec)
    {
        return std::sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    } 

    __device__ __host__ Glpa::GPU_VEC_3D normalize(Glpa::GPU_VEC_3D vec)
    {
        float length = getLength(vec);
        return Glpa::GPU_VEC_3D(vec.x / length, vec.y / length, vec.z / length);
    }

    __device__ __host__ float cos(Glpa::GPU_VEC_3D vec1, Glpa::GPU_VEC_3D vec2)
    {
        return dot(vec1, vec2) / (getLength(vec1) * getLength(vec2));
    }

} GPU_VECTOR_MG;

}

#endif GLPA_VECTOR_H_

