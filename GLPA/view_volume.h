#ifndef VIEW_VOLUME_H_
#define VIEW_VOLUME_H_

#include <vector>
#include <math.h>

#include "cg.h"
#include "file.h"

class ViewVolume
{
public :
    ViewVolume()
    {
        point3D.resize(8);
        pointXZ.resize(4);
        pointYZ.resize(4);

        face.resize(6);

        lineStartPoint.resize(12);
        lineEndPoint.resize(12);
        lineVec.resize(12);
    }

    void define
    (
        double near_screen_z, double far_screen_z,
        SIZE2 near_screen_pixel_size, SIZE2 far_screen_pixel_size,
        ANGLE* view_angle, VECTOR2D aspect_ratio
    );

    // Clip rectangular range and view volume   
    int clip(std::vector<OBJ_FILE> object_data, std::vector<double> range_degree, ANGLE view_angle, int loop_i);

private :
    std::vector<VECTOR3D> point3D;
    std::vector<VECTOR_XZ> pointXZ;
    std::vector<VECTOR_YZ> pointYZ;

    std::vector<CALCFACE> face;

    std::vector<VECTOR3D> lineStartPoint;
    std::vector<VECTOR3D> lineEndPoint;
    std::vector<VECTOR3D> lineVec;

    VECTOR vec;
};

#endif VIEW_VOLUME_H_
