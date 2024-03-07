/**
 * @file glpa_.h
 * @brief
 * 日本語 : GLPAで使用している構造体やマクロをまとめたもの。このファイルはGLPAのいずれかのファイルもインクルードしない。
 * English : A collection of structures and macros used in glpa. This file does not include any of the glpa files.
 * @author Tanso
 * @date 2024-1
*/


#ifndef CG_H_
#define CG_H_

#include <vector>
#include <string>



/**********************************************************************************************************************
 * 日本語 : 各データの型の初期値を記述。
 * English : Describes the initial value of each data type.
**********************************************************************************************************************/

#define GLPA_WSTRING_DEF L"NULL"


/**********************************************************************************************************************
 * 日本語 : 数学に関連する構造体やマクロを以下に記述。
 * English : Structures and macros related to mathematics are described below.
**********************************************************************************************************************/


/********************************************************************************
 * 日本語 : ベクトルに関連するものらを記述。
 * English : Describes those related to vectors.
********************************************************************************/

typedef struct tagVECTOR2D{
    double x;
    double y;
} Vec2d;

typedef struct tagVECTOR2DXZ{
    double x;
    double z;
} Vec2dXZ;

typedef struct tagVECTOR2DYZ{
    double y;
    double z;
} Vec2dYZ;


typedef struct tagVECTOR3D{
    double x;
    double y;
    double z;
} Vec3d;


typedef struct tagNUMCOMB3{
    int n1; // number 1
    int n2; // number 2
    int n3; // number 3
} NumComb3;


#define PI 3.14159265

#define RAD(angle) \
    angle *PI /180



/********************************************************************************
 * 日本語 : 図形を表すために使用するものらを記述。
 * English : Describes the objects used to represent the figure.
********************************************************************************/


typedef struct tagShapeLine{
    Vec3d startV;
    Vec3d endV;
    Vec3d vec;
} shapeLine;


typedef struct tagFaceNormals{
    std::vector<Vec3d> v;
    std::vector<Vec3d> normal;
} FaceNormals;


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

#define RECT_L1_STARTV RECT_FRONT_TOP_LEFT
#define RECT_L1_ENDV RECT_FRONT_TOP_RIGHT
#define RECT_L2_STARTV RECT_FRONT_TOP_RIGHT
#define RECT_L2_ENDV RECT_FRONT_BOTTOM_RIGHT
#define RECT_L3_STARTV RECT_FRONT_BOTTOM_RIGHT
#define RECT_L3_ENDV RECT_FRONT_BOTTOM_LEFT
#define RECT_L4_STARTV RECT_FRONT_BOTTOM_LEFT
#define RECT_L4_ENDV RECT_FRONT_TOP_LEFT

#define RECT_L5_STARTV RECT_FRONT_TOP_LEFT
#define RECT_L5_ENDV RECT_BACK_TOP_LEFT
#define RECT_L6_STARTV RECT_FRONT_TOP_RIGHT
#define RECT_L6_ENDV RECT_BACK_TOP_RIGHT
#define RECT_L7_STARTV RECT_FRONT_BOTTOM_RIGHT
#define RECT_L7_ENDV RECT_BACK_BOTTOM_RIGHT
#define RECT_L8_STARTV RECT_FRONT_BOTTOM_LEFT
#define RECT_L8_ENDV RECT_BACK_BOTTOM_LEFT

#define RECT_L9_STARTV RECT_BACK_TOP_LEFT
#define RECT_L9_ENDV RECT_BACK_TOP_RIGHT
#define RECT_L10_STARTV RECT_BACK_TOP_RIGHT
#define RECT_L10_ENDV RECT_BACK_BOTTOM_RIGHT
#define RECT_L11_STARTV RECT_BACK_BOTTOM_RIGHT
#define RECT_L11_ENDV RECT_BACK_BOTTOM_LEFT
#define RECT_L12_STARTV RECT_BACK_BOTTOM_LEFT
#define RECT_L12_ENDV RECT_BACK_TOP_LEFT


/**********************************************************************************************************************
 * 日本語 : メッシュデータから読み込む情報を格納するために使用するものらを記述。
 * English : Describes what is used to store information to be read from the mesh data.
**********************************************************************************************************************/

/**
 * 日本語 : メッシュの頂点を格納する。それぞれのパラメーターでインデックス番号がそのまま頂点番号になる。
 * English : Stores the vertices of the mesh. The index number in each parameter is the vertex number as it is.
*/
typedef struct tagVertices{
    std::vector<Vec3d> world;
    std::vector<Vec2d> uv;
    std::vector<Vec3d> normal;
} Vertices; 

/**
 * 日本語 : ポリゴンを構成する3頂点の頂点番号を格納する。
 * English : Stores the vertex numbers of the three vertices that make up the polygon.
*/
typedef struct tagPolygons{
    std::vector<NumComb3> vId;
    std::vector<NumComb3> uvId;
    std::vector<NumComb3> normalId;
} Polygons;

/**
 * 日本語 : メッシュやポリゴンの囲む直方体のデータを格納する。
 * English : Stores the data of the rectangle that encloses the mesh and polygons.
*/
typedef struct tagRangeRect{
    bool status = false;
    Vec3d origin;
    Vec3d opposite;
    std::vector<Vec3d> wVertex;
} RangeRect;



/**********************************************************************************************************************
 * 日本語 : レンダリング処理で使用し、対象のデータの識別するものらを記述。
 * English : Describes the data to be used in the rendering process and identifies the target data.
**********************************************************************************************************************/


/**
 * 日本語 : ポリゴンの名前を保存する際に、どのオブジェクトどのメッシュどのポリゴンかも保存するために使用する。
 * English : Used to save the polygon name, which object, which mesh, and which polygon.
*/
typedef struct tagPolyNameInfo{
    std::wstring objName;
    int polyId;
} PolyNameInfo;


typedef struct tagMultiSidedShape{
    std::vector<Vec3d> wVs;
    std::vector<Vec2d> vs;
} MultiSidedShape;


typedef struct tagRasterizeSource{
    PolyNameInfo  renderPoly;
    std::vector<Vec3d> polyCamVs;
    Vec3d polyN;

    MultiSidedShape scPixelVs;
    
} RasterizeSource;


typedef struct tagDebugSt{
    std::wstring objName;
    int polyId;
    Vec3d inxtn;
} DebugSt;



#endif CG_H_