/**
 * @file glpa_macro.h
 * @brief
 * 日本語 : GLPAで使用している構造体やマクロをまとめたもの。このファイルはGLPAのいずれかのファイルもインクルードしない。
 * English : A collection of structures and macros used in glpa. This file does not include any of the glpa files.
 * @author Tanso
 * @date 2024-1
*/


#ifndef GLPA_MACRO_CG_H_
#define GLPA_MACRO_CG_H_

#include <vector>


/**********************************************************************************************************************
 * 日本語 : 数学に関連する構造体やマクロを以下に記述。
 * English : Structures and macros related to mathematics are described below.
**********************************************************************************************************************/


/********************************************************************************
 * 日本語 : ベクトルに関連するものらを記述。
 * English : Describes those related to vectors.
********************************************************************************/

typedef struct tagVECTOR2D
{
    double x;
    double y;
} VECTOR2D;

typedef struct tagVECTOR3D
{
    double x;
    double y;
    double z;
} VECTOR3D;



typedef struct tagNUMCOMB3
{
    int num1; // number 1
    int num2; // number 2
    int num3; // number 3
} NUMCOMB3;


/**********************************************************************************************************************
 * 日本語 : メッシュデータから読み込む情報を格納するために使用するものらを記述。
 * English : Describes what is used to store information to be read from the mesh data.
**********************************************************************************************************************/


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


#endif GLPA_MACRO_CG_H_