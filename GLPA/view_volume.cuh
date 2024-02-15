#ifndef VIEW_VOLUME_H_
#define VIEW_VOLUME_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <math.h>

#include "cg.h"

/// @brief Within 3DCG, it has data related to the view volume.
class ViewVolume
{
public :
    ViewVolume(){
        xzV.resize(4);
        yzV.resize(4);
        v.resize(8);

        lines.resize(12);
        face.resize(6);
    };

    std::vector<Vec2dXZ> xzV;
    std::vector<Vec2dYZ> yzV;
    std::vector<Vec3d> v;

    std::vector<shapeLine> lines;
    std::vector<FaceNormal> face;
};

#endif VIEW_VOLUME_H_