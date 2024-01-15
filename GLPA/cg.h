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


/**********************************************************************************************************************
 * 日本語 : メッシュデータから読み込む情報を格納するために使用するものらを記述。
 * English : Describes what is used to store information to be read from the mesh data.
**********************************************************************************************************************/

/**
 * 日本語 : メッシュの頂点を格納する。それぞれのパラメーターでインデックス番号がそのまま頂点番号になる。
 * English : Stores the vertices of the mesh. The index number in each parameter is the vertex number as it is.
*/
typedef struct tagVERTEX{
    std::vector<Vec3d> world;
    std::vector<Vec2d> uv;
    std::vector<Vec3d> normal;
} Vertex;

/**
 * 日本語 : ポリゴンを構成する3頂点の頂点番号を格納する。
 * English : Stores the vertex numbers of the three vertices that make up the polygon.
*/
typedef struct tagPOLYGON{
    std::vector<NumComb3> vId;
    std::vector<NumComb3> uvId;
    std::vector<NumComb3> normalId;
} Polygons;

/**
 * 日本語 : メッシュやポリゴンの囲む直方体のデータを格納する。
 * English : Stores the data of the rectangle that encloses the mesh and polygons.
*/
typedef struct tagRANGE_RECT{
    bool status = false;
    Vec3d origin;
    Vec3d opposite;
    std::vector<Vec3d> wVertex;
} RangeRect;


#endif CG_H_