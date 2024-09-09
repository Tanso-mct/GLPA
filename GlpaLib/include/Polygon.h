#ifndef GLPA_POLYGON_H_
#define GLPA_POLYGON_H_

#include <vector>
#include <string>

#include "Vector.cuh"
#include "Matrix.cuh"


namespace Glpa
{

typedef struct _GPU_POLYGON
{
    Glpa::GPU_VEC_3D wv[3];
    Glpa::GPU_VEC_2D uv[3];
    Glpa::GPU_VEC_3D n;

    __device__ __host__ _GPU_POLYGON(){}

    __device__ __host__ _GPU_POLYGON(_GPU_POLYGON sPoly, Glpa::GPU_MAT_4X4& mtTransRot, Glpa::_GPU_MAT_4X4& mtRot)
    {
        for (int i = 0; i < 3; i++)
        {
            wv[i] = mtTransRot.productLeft3x1(sPoly.wv[i]);
            uv[i] = sPoly.uv[i];
        }

        n = mtRot.productLeft3x1(sPoly.n);
    }

    __device__ __host__ GPU_BOOL isFacing(Glpa::GPU_MAT_4X4& mtTransRot, Glpa::_GPU_MAT_4X4& mtRot)
    {
        Glpa::GPU_VEC_3D cnvtV = mtTransRot.productLeft3x1(wv[0]);
        Glpa::GPU_VEC_3D cnvtN = mtRot.productLeft3x1(n);

        Glpa::GPU_VECTOR_MG vecMg;
        float dot = vecMg.dot(cnvtV, cnvtN);

        GPU_IF(dot <= 0, branch2)
        {
            return TRUE;
        }

        return FALSE;
    }

} GPU_POLYGON;

class Polygon
{
private :
    std::vector<int> wvI;
    std::vector<int> uvI;
    Glpa::Vec3d normal;

public :
    Polygon();
    ~Polygon();

    void addV(int argWvI, int argUvI);
    void setNormal(Glpa::Vec3d argNormal){normal = argNormal;};

    Glpa::GPU_POLYGON getData(std::vector<Glpa::Vec3d*>& wv, std::vector<Glpa::Vec2d*>& uv);
};

}


#endif GLPA_POLYGON_H_