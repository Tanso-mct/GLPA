#ifndef CAMERA_H_
#define CAMERA_H_

#include <vector>
#include <math.h>
#include <windows.h>
#include <stdio.h>

#include "cgmath.cuh"
#include "file.h"
#include "graphic.h"

#define VP1 0
#define VP2 1
#define VP3 2
#define VP4 3

class CAMERA
{
public :
    VECTOR3D wPos;
    VECTOR3D rotAngle;

    bool initialized = false;

    double nearZ;
    double farZ;
    double viewAngle;
    VECTOR2D aspectRatio;

    SIZE2 nearScreenSize;
    SIZE2 farScreenSize;
    
    std::vector<VECTOR3D> viewPoint;
    std::vector<VECTOR_XZ> viewPointXZ;
    std::vector<VECTOR_YZ> viewPointYZ;

    std::vector<VECTOR3D> viewVolumeFaceVertex;
    std::vector<VECTOR3D> viewVolumeFaceNormal;

    MATRIX mtx;
    VECTOR vec;
    EQUATION eq;

    std::vector<int> withinRangeAryNum;
    std::vector<INT2D> numPolyFacing;
    std::vector<VECTOR3D> polyVertex;
    std::vector<INT2D> numPolyInViewVolume;
    std::vector<INT2D> numPolyNotInViewVolume;
    std::vector<INT2D> numPolyAllVINotInViewVolume;


    void initialize(); // Initialize data
    void defViewVolume(); // define clipping area

    // Range coordinate transformation
    void coordinateTransRange(std::vector<OBJ_FILE>* objData);

    void clippingRange(std::vector<OBJ_FILE> objData);

    // Determining whether the face is front or back
    void polyBilateralJudge(std::vector<OBJ_FILE> objData);

    // Coordinate transformation of the vertices of the surface to be drawn
    void coordinateTransV(std::vector<OBJ_FILE> objData);

    // Determine if a point is on the face of a specific plane and on a specific line segment
    bool confirmI
    (
        int exits_Idata,
        double left_GreaterThan1Data,  double right_GreaterThan1Data,
        double left_LessThan1Data,  double right_LessThan1Data,
        double left_GreaterThan2Data,  double right_GreaterThan2Data,
        double left_LessThan2Data,  double right_LessThan2Data,
        VECTOR3D line_plane_I, VECTOR3D line_vertex_A, VECTOR3D line_vertex_B,
        int withInRangeAryNumd_Oata, int numPolyfacing_Data
    );

    // Determines if a vertex is in the view volume
    bool vertexInViewVolume(VECTOR3D vertex);
    
    std::vector<VECTOR3D> usedLineVA;
    std::vector<VECTOR3D> usedLineVB;
    // Determine if polygon is in view volume and store array number
    void polyInViewVolumeJudge(std::vector<OBJ_FILE> objData);
};

#endif CAMERA_H_
