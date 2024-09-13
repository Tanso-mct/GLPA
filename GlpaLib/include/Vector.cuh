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

    __device__ __host__ void set(float argX, float argY)
    {
        x = argX;
        y = argY;
    }

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

    Vec2d operator=(const std::initializer_list<float>& values)
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

    __device__ __host__ void set(float argX, float argY, float argZ)
    {
        x = argX;
        y = argY;
        z = argZ;
    }

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

    __device__ __host__ void operator+=(const _GPU_VEC_3D& other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
    }

    __device__ __host__ void operator-=(const _GPU_VEC_3D& other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;
    }

    __device__ __host__ void operator*=(float scalar)
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
    }

    __device__ __host__ void operator/=(float scalar)
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

typedef struct _GPU_VECTOR_MANAGER
{
    // Vector 2D
    __device__ __host__ Glpa::GPU_VEC_2D getVec(Glpa::GPU_VEC_2D start, Glpa::GPU_VEC_2D end)
    {
        return Glpa::GPU_VEC_2D(end.x - start.x, end.y - start.y);
    }

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
    __device__ __host__ Glpa::GPU_VEC_3D getVec(Glpa::GPU_VEC_3D start, Glpa::GPU_VEC_3D end)
    {
        return Glpa::GPU_VEC_3D(end.x - start.x, end.y - start.y, end.z - start.z);
    }

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

typedef struct _GPU_LINE_3D
{
    Glpa::GPU_VEC_3D start;
    Glpa::GPU_VEC_3D end;
    Glpa::GPU_VEC_3D vec;

    __device__ __host__ _GPU_LINE_3D(){};

    __device__ __host__ _GPU_LINE_3D(Glpa::GPU_VEC_3D argStart, Glpa::GPU_VEC_3D argEnd)
    {
        start = argStart;
        end = argEnd;
        vec = end - start;
    }

    __device__ __host__ void set(Glpa::GPU_VEC_3D argStart, Glpa::GPU_VEC_3D argEnd)
    {
        start = argStart;
        end = argEnd;
        vec = end - start;
    }

    __device__ __host__ float getLength() const
    {
        return std::sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    }

    __device__ __host__ Glpa::GPU_VEC_3D getPoint(float t) const
    {
        return start + vec * t;
    }

    __device__ __host__ GPU_BOOL operator==(const _GPU_LINE_3D& other) const
    {
        return (start == other.start && end == other.end) ? TRUE : FALSE;
    }

    __device__ __host__ GPU_BOOL operator!=(const _GPU_LINE_3D& other) const
    {
        return (!(*this == other)) ? TRUE : FALSE;
    }

    __device__ __host__ _GPU_LINE_3D operator=(const _GPU_LINE_3D& other)
    {
        start = other.start;
        end = other.end;
        return *this;
    }

} GPU_LINE_3D;

typedef struct _GPU_FACE_3D
{
    int type = GPU_IS_EMPTY;
    Glpa::GPU_VEC_3D v;
    Glpa::GPU_VEC_3D n;

    Glpa::GPU_VEC_3D vs[GPU_MAX_FACE_V];

    __device__ __host__ _GPU_FACE_3D(){};

    __device__ __host__ _GPU_FACE_3D(Glpa::GPU_VEC_3D argV, Glpa::GPU_VEC_3D argN)
    {
        v = argV;
        n = argN;
    };

    __device__ __host__ void set(Glpa::GPU_VEC_3D argV, Glpa::GPU_VEC_3D vecA, Glpa::GPU_VEC_3D vecB)
    {
        v = argV;

        n.set
        (
            vecA.y * vecB.z - vecA.z * vecB.y,
            vecA.z * vecB.x - vecA.x * vecB.z,
            vecA.x * vecB.y - vecA.y * vecB.x
        );
    }

    __device__ __host__ void setTriangle(Glpa::GPU_VEC_3D v0, Glpa::GPU_VEC_3D v1, Glpa::GPU_VEC_3D v2)
    {
        type = GPU_IS_TRIANGLE;
        vs[0] = v0;
        vs[1] = v1;
        vs[2] = v2;
    }

    __device__ __host__ void setSquare(Glpa::GPU_VEC_3D v0, Glpa::GPU_VEC_3D v1, Glpa::GPU_VEC_3D v2, Glpa::GPU_VEC_3D v3)
    {
        type = GPU_IS_SQUARE;
        vs[0] = v0;
        vs[1] = v1;
        vs[2] = v2;
        vs[3] = v3;
    }

    __device__ __host__ GPU_BOOL isInside(Glpa::GPU_VEC_3D& point)
    {
        Glpa::GPU_VECTOR_MG vecMgr;
        GPU_IF(type == GPU_IS_TRIANGLE, br2)
        {
            // Dot product of a polygon line segment and a point.
            // The polygon line segment is pla, and the point is p.
            float dotPlaP[3] = 
            {
                vecMgr.cos(vecMgr.getVec(vs[0], vs[1]), vecMgr.getVec(vs[0], point)),
                vecMgr.cos(vecMgr.getVec(vs[1], vs[2]), vecMgr.getVec(vs[1], point)),
                vecMgr.cos(vecMgr.getVec(vs[2], vs[0]), vecMgr.getVec(vs[2], point))
            };

            // Let plb be the line segment from the polygon vertex that is not pla.
            float dotPlaPlb[3] = 
            {
                vecMgr.cos(vecMgr.getVec(vs[0], vs[1]), vecMgr.getVec(vs[0], vs[2])),
                vecMgr.cos(vecMgr.getVec(vs[1], vs[2]), vecMgr.getVec(vs[1], vs[0])),
                vecMgr.cos(vecMgr.getVec(vs[2], vs[0]), vecMgr.getVec(vs[2], vs[1]))
            };

            // PLAとPLBの内積の値より、PLAとPの内積の値が大きいかどうかを判定する。
            GPU_BOOL isDotBigger = TRUE;
            for (int i = 0; i < 3; i++)
            {
                isDotBigger *= GPU_CO(dotPlaP[i] >= dotPlaPlb[i], TRUE, FALSE);
            }

            // If all three test results are true, the point is inside the triangle.
            return isDotBigger;
        }

        GPU_IF(type == GPU_IS_SQUARE, br2)
        {
            // Dot product of a polygon line segment and a point.
            // The polygon line segment is pla, and the point is p.
            float dotPlaP[4] = 
            {
                vecMgr.cos(vecMgr.getVec(vs[0], vs[1]), vecMgr.getVec(vs[0], point)),
                vecMgr.cos(vecMgr.getVec(vs[1], vs[2]), vecMgr.getVec(vs[1], point)),
                vecMgr.cos(vecMgr.getVec(vs[2], vs[3]), vecMgr.getVec(vs[2], point)),
                vecMgr.cos(vecMgr.getVec(vs[3], vs[0]), vecMgr.getVec(vs[3], point))
            };

            // Let plb be the line segment from the polygon vertex that is not pla.
            float dotPlaPlb[4] = 
            {
                vecMgr.cos(vecMgr.getVec(vs[0], vs[1]), vecMgr.getVec(vs[0], vs[2])),
                vecMgr.cos(vecMgr.getVec(vs[1], vs[2]), vecMgr.getVec(vs[1], vs[0])),
                vecMgr.cos(vecMgr.getVec(vs[2], vs[3]), vecMgr.getVec(vs[2], vs[1])),
                vecMgr.cos(vecMgr.getVec(vs[3], vs[0]), vecMgr.getVec(vs[3], vs[2]))
            };

            // PLAとPLBの内積の値より、PLAとPの内積の値が大きいかどうかを判定する。
            GPU_BOOL isDotBigger = TRUE;
            for (int i = 0; i < 4; i++)
            {
                isDotBigger *= GPU_CO(dotPlaP[i] >= dotPlaPlb[i], TRUE, FALSE);
            }

            // If all three test results are true, the point is inside the triangle.
            return isDotBigger;
        }

        return FALSE;
    }

} GPU_FACE_3D;

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

}

#endif GLPA_VECTOR_H_

