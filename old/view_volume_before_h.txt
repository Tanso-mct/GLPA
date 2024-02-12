#ifndef VIEW_VOLUME_H_
#define VIEW_VOLUME_H_

#include <vector>
#include <math.h>

// #include "cg.h"
#include "file.h"

/// @brief Within 3DCG, it has data related to the view volume.
class ViewVolume
{
public :
    ViewVolume()
    {
        point3D.resize(RECTVS);
        pointXZ.resize(SQUAREVS);
        pointYZ.resize(SQUAREVS);

        face.resize(RECTSURFACES);

        lineStartPoint.resize(RECTLINES);
        lineEndPoint.resize(RECTLINES);
        lineVec.resize(RECTLINES);
    }

    std::vector<CALCFACE> face;

    /// @brief Define the view volume range. Required whenever the view angle changes.
    /// @param near_screen_z z-value of near screen surface.
    /// @param far_screen_z z-value of far screen surface.
    /// @param near_screen_size Size of the near screen surface in three dimensions, with width and height data.
    /// @param far_screen_size Size of the far screen surface in three dimensions, with width and height data.
    /// @param view_angle Camera viewing angle.
    /// @param aspect_ratio Camera screen aspect ratio.
    void define
    (
        double near_screen_z, double far_screen_z,
        SIZE2* near_screen_size, SIZE2* far_screen_size,
        ANGLE* view_angle, VECTOR2D aspect_ratio
    );

    /// @brief Clip rectangular range and view volume   
    /// @param mesh_data Loaded 3D mesh data. 
    /// @param range_degree Angle from the origin of the origin and opposite of the rectangular range.
    /// @param view_angle Camera viewing angle.
    /// @param loop_i Loop counter variable when a mesh thing is looping.
    /// @return If the mesh indicated by the loop counter variable is in the view volume, the loop counter variable 
    /// is returned. otherwise, NULL_INDEX(-1) is returned.
    int clipRange(std::vector<OBJ_FILE> mesh_data, std::vector<double> range_degree, ANGLE view_angle, int loop_i);

    /// @brief Clip rectangular range and view volume   
    /// @param range_rect 
    /// @param range_degree 
    /// @param view_angle 
    /// @param loop_i 
    /// @return 
    int clipRange
    (
        std::vector<RANGE_RECT> range_rect, std::vector<double> range_degree, ANGLE view_angle, int loop_i
    );

    /// @brief Clip vertex and view volume
    /// @param vertex_z z-value of the vertex to be calculated.
    /// @param vertex_degree_xz Angle of the vertex in the XZ axis from the origin
    /// @param vertex_degree_yz Angle from the origin of the vertex on the YZ axis
    /// @param view_angle Camera viewing angle.
    /// @param loop_i Loop counter variable when a mesh thing is looping.
    /// @return Returns true if the vertex exists in the view volume, false if not.
    bool clipV(double vertex_z, double vertex_degree_xz, double vertex_degree_zy, ANGLE view_angle, int loop_i);

private :
    std::vector<VECTOR3D> point3D;
    std::vector<VECTOR_XZ> pointXZ;
    std::vector<VECTOR_YZ> pointYZ;

    std::vector<VECTOR3D> lineStartPoint;
    std::vector<VECTOR3D> lineEndPoint;
    std::vector<VECTOR3D> lineVec;

    VECTOR vec;
};

#endif VIEW_VOLUME_H_