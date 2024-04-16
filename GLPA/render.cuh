#ifndef RENDER_CUH_
#define RENDER_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "object.h"
#include "camera.cuh"

#include "cg.h"
#include "error.h"

#include "matrix.cuh"
#include "vector.cuh"

__global__ void glpaGpuPrepareObj(
    int object_size,
    float* object_world_vs,
    float* matrix_camera_transformation_and_rotation,
    float camera_near_z,
    float camera_far_z,
    float* camera_view_angle_cos,
    int* result_object_in_view_volume_judge_ary
);

__device__ int judgePolyVInViewVolume(
    float* converted_poly_v,
    float camera_far_z,
    float camera_near_z,
    float* camera_view_angle
);


__global__ void glpaGpuRender(
    float* poly_vertecies,
    float* poly_normals,
    int poly_amount,
    float* matrix_camera_transformation_and_rotation,
    float* matrix_camera_rotation,
    float camera_far_z,
    float camera_near_z,
    float* camera_view_angle
);


class Render{
public :
    void prepareObjs(
        std::unordered_map<std::wstring, Object> source_objects,
        Camera cam
    );

    void render(
        std::unordered_map<std::wstring, Object> source_objects,
        Camera cam,
        LPDWORD buffer
    );

private :
    float* hMtCamTransRot;
    float* hMtCamRot;

    float* hCamViewAngleCos;
    int* hObjInJudgeAry;

};


#endif RENDER_CUH_