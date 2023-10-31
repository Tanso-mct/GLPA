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

typedef struct tagANGLE
{
    double vert;
    double horiz;
} ANGLE;

typedef struct tagCALCFACE
{
    VECTOR3D oneV;
    VECTOR3D normal;
} CALCFACE;

typedef struct tagPOLYINFO
{
    int relRenderingSourceSt;
    int meshID;
    int polyID;
    std::vector<VECTOR3D> lineStartPoint;
    std::vector<VECTOR3D> lineEndPoint;
    std::vector<VECTOR3D> lineVec;
    VECTOR3D polyNormal;
    std::vector<int> viewVoluneIExistID;
    std::vector<VECTOR3D> vecI;
} POLYINFO;

typedef struct tagSMALL_POLYINFO
{
    int meshID;
    int polyID;
    VECTOR3D oneV;
    VECTOR3D nomarl;
} SMALL_POLYINFO;

typedef struct tagRENDERSOURCE
{
    int meshID;
    int polyID;
    std::vector<VECTOR3D> polyV;
    std::vector<VECTOR3D> iV;
    std::vector<VECTOR2D> scrPolyV;
    std::vector<VECTOR2D> scrIV;
} RENDERSOUCE;

typedef struct tagMESHINFO
{
    int meshID;
    std::vector<int> facingPolyID;
} MESHINFO;

class VIEWVOLUME
{
public :
    std::vector<VECTOR3D> point3D;
    std::vector<VECTOR_XZ> pointXZ;
    std::vector<VECTOR_YZ> pointYZ;

    std::vector<CALCFACE> face;

    std::vector<VECTOR3D> lineStartPoint;
    std::vector<VECTOR3D> lineEndPoint;
    std::vector<VECTOR3D> lineVec;

    VECTOR vec;

    VIEWVOLUME()
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
};

class CAMERA
{
public :
    VECTOR3D wPos;
    VECTOR3D rotAngle;

    double nearZ;
    double farZ;
    ANGLE viewAngle;
    VECTOR2D aspectRatio;

    SIZE2 nearScrSize;
    SIZE2 farScrSize;

    VIEWVOLUME viewVolume;
    
    std::vector<MESHINFO> meshInfo;

    std::vector<POLYINFO> sourcePolyInfo;
    std::vector<POLYINFO> calcPolyInfo;
    std::vector<POLYINFO> searchPolyInfo;

    std::vector<RENDERSOUCE> renderSouce;

    MATRIX mtx;
    VECTOR vec;
    EQUATION eq;
    TRIANGLE_RATIO tri;

    CAMERA()
    {
        wPos = {0, 0, 0};
        rotAngle = {0, 0, 0};

        nearZ = 1;
        farZ = 10000;
        viewAngle.horiz = 80;
        aspectRatio = {16, 9};
    }

    // Rect Range coordinate transformation
    void coordinateTransRectRange(std::vector<OBJ_FILE>* meshData);

    // Determination of intersection of OBJECT with view volume
    std::vector<std::vector<SMALL_POLYINFO>> clippingMeshRectRange(std::vector<OBJ_FILE> meshData);

    // Determining whether the face is front or back
    void polyBilateralJudge
    (
        std::vector<OBJ_FILE>* meshData,
        std::vector<std::vector<SMALL_POLYINFO>> in_view_volume_poly_data
    );

    // bool initialized = false;


    // std::vector<int> inViewVolumeMeshID;
    
    
    // std::vector<VECTOR3D> viewPoint;
    // std::vector<VECTOR_XZ> viewPointXZ;
    // std::vector<VECTOR_YZ> viewPointYZ;

    // std::vector<VECTOR3D> viewVolumeFaceVertex;
    // std::vector<VECTOR3D> viewVolumeFaceNormal;

    // std::vector<int> withinRangeAryNum;
    // std::vector<std::vector<int>> numPolyFacing;

    // std::vector<VECTOR3D> polyVertex;
    // std::vector<VECTOR3D> polyNormal;

    // std::vector<std::vector<int>> numPolyInViewVolume;

    // Polygon clipped vertices in view volume
    // The index is numPolyInViewVolume
    // std::vector<std::vector<std::vector<VECTOR3D>>> clippedPolyVertex;

    // Stores polygons where possible intersections between polygon line segments and view volume surfaces exist
    // std::vector<std::vector<int>> numPolyExitsIViewVolume;

    // Stores polygons with intersections with the view volume plane
    // std::vector<std::vector<int>> numPolyTrueIViewVolume;

    // Store polygons that are outside of the view volume at all three points and all three sides
    // std::vector<std::vector<int>> numPolyAllVLINENotInViewVolume;

    // void defViewVolume(); // define clipping area

    // Coordinate transformation of the vertices of the surface to be drawn
    void coordinateTrans(std::vector<OBJ_FILE> meshData);

    // Determines if a vertex is in the view volume
    std::vector<bool> vertexInViewVolume(std::vector<VECTOR3D> vertex);
    
    // Determine if polygon is in view volume and store array number
    void polyInViewVolumeJudge(std::vector<OBJ_FILE> meshData);

    // Intersection judgment between polygon and view volume
    std::vector<std::vector<int>> clippingRange(std::vector<std::vector<RANGE_CUBE_POLY>> range_polygon, int process_object_amout);

};

#endif CAMERA_H_
