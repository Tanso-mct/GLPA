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

#include "vector.cuh"
#include "matrix.cuh"


__global__ void glpaGpuPrepareObj(
    int object_size,
    float* object_world_vs,
    float* matrix_camera_transformation_and_rotation,
    float camera_near_z,
    float camera_far_z,
    float* camera_view_angle_cos,
    int* result_object_in_view_volume_judge_ary
);


__global__ void glpaGpuRender(
    float* poly_vertices,
    float* poly_normals,
    int poly_amount,
    float* matrix_camera_transformation_and_rotation,
    float* matrix_camera_rotation,
    float camera_far_z,
    float camera_near_z,
    float* camera_view_angle,
    float* view_volume_vertices,
    float* view_volume_normals,
    float* near_screen_size,
    float* screen_pixel_size
);


class Render{
public :
    Render();

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
    std::vector<float> hMtCamTransRot;
    std::vector<float> hMtCamRot;

    std::vector<float> hCamViewAngleCos;

    int* hObjInJudgeAry;

};


#endif RENDER_CUH_