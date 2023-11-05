#ifndef CG_H_
#define CG_H_

#include <vector>
#include "cgmath.cuh"

/** @def
 * In a sequence, XYZ.
 */
#define VX 0
#define VY 1
#define VZ 2

/** @def
 * Indicates the number of polygon vertices.
 */
#define POLYV 3

/** @def
 * The number of vertices of the rectangle.
 */
#define RECTV 8

/** @def
 * The vertex number when viewing the view volume in each of the XZ and ZY axes. The order is clockwise from the bottom 
 * left vertex.
 */
#define VP1 0
#define VP2 1
#define VP3 2
#define VP4 3

#define SURFACE_TOP 0
#define SURFACE_FRONT 1
#define SURFACE_RIGHT 2
#define SURFACE_LEFT 3
#define SURFACE_BACK 4
#define SURFACE_BOTTOM 5

#define RECT_FRONT_TOP_LEFT 0
#define RECT_FRONT_TOP_RIGHT 1
#define RECT_FRONT_BOTTOM_RIGHT 2
#define RECT_FRONT_BOTTOM_LEFT 3
#define RECT_BACK_TOP_LEFT 4
#define RECT_BACK_TOP_RIGHT 5
#define RECT_BACK_BOTTOM_RIGHT 6
#define RECT_BACK_BOTTOM_LEFT 7

#define RECT_L1_STARTPT RECT_FRONT_TOP_LEFT
#define RECT_L1_ENDPT RECT_FRONT_TOP_RIGHT
#define RECT_L2_STARTPT RECT_FRONT_TOP_RIGHT
#define RECT_L2_ENDPT RECT_FRONT_BOTTOM_RIGHT
#define RECT_L3_STARTPT RECT_FRONT_BOTTOM_RIGHT
#define RECT_L3_ENDPT RECT_FRONT_BOTTOM_LEFT
#define RECT_L4_STARTPT RECT_FRONT_BOTTOM_LEFT
#define RECT_L4_ENDPT RECT_FRONT_TOP_LEFT

#define RECT_L5_STARTPT RECT_FRONT_TOP_LEFT
#define RECT_L5_ENDPT RECT_BACK_TOP_LEFT
#define RECT_L6_STARTPT RECT_FRONT_TOP_RIGHT
#define RECT_L6_ENDPT RECT_BACK_TOP_RIGHT
#define RECT_L7_STARTPT RECT_FRONT_BOTTOM_RIGHT
#define RECT_L7_ENDPT RECT_BACK_BOTTOM_RIGHT
#define RECT_L8_STARTPT RECT_FRONT_BOTTOM_LEFT
#define RECT_L8_ENDPT RECT_BACK_BOTTOM_LEFT

#define RECT_L9_STARTPT RECT_BACK_TOP_LEFT
#define RECT_L9_ENDPT RECT_BACK_TOP_RIGHT
#define RECT_L10_STARTPT RECT_BACK_TOP_RIGHT
#define RECT_L10_ENDPT RECT_BACK_BOTTOM_RIGHT
#define RECT_L11_STARTPT RECT_BACK_BOTTOM_RIGHT
#define RECT_L11_ENDPT RECT_BACK_BOTTOM_LEFT
#define RECT_L12_STARTPT RECT_BACK_BOTTOM_LEFT
#define RECT_L12_ENDPT RECT_BACK_TOP_LEFT

#define RECT_L1 0
#define RECT_L2 1
#define RECT_L3 2
#define RECT_L4 3
#define RECT_L5 4
#define RECT_L6 5
#define RECT_L7 6
#define RECT_L8 7
#define RECT_L9 8
#define RECT_L10 9
#define RECT_L11 10
#define RECT_L12 11

/**
 * @struct NUMCOMB3
 * @brief An int type integer with three values. Created for use with vector type variables.
**/
typedef struct tagNUMCOMB3
{
    int num1; // number 1
    int num2; // number 2
    int num3; // number 3
} NUMCOMB3;

typedef struct tagRGBA
{
    int r;
    int g;
    int b;
    int a;
} RGBA;

typedef struct tagIMAGE
{
    int width;
    int height;
    int colorDepth;
    int compType;
    int format;
    std::vector<RGBA> rgbaData;
} IMAGE;

typedef struct tagANGLE
{
    double vert;
    double horiz;
} ANGLE;

typedef struct tagVERTEX
{
    std::vector<VECTOR3D> world; // world coordinate
    std::vector<VECTOR2D> uv; // uv coordinate
    std::vector<VECTOR3D> normal; // normal
} VERTEX;

typedef struct tagPOLYGON
{
    std::vector<NUMCOMB3> v; // vertex number
    std::vector<NUMCOMB3> uv; // uv number
    std::vector<NUMCOMB3> normal; // normal number
} POLYGON;

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
    std::vector<int> meshID;
    std::vector<int> polyID;
    std::vector<VECTOR3D> oneV;
    std::vector<VECTOR3D> normal;
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

#endif CG_H_