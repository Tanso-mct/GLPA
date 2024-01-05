/**
 * @file glpa_macro.h
 * @brief A list of macros used in glpa. 
 * It prevents errors due to mutual inclusion and at the same time allows you to check the macros used in glpa.
 * @author Tanso
 * @date Date?iyear-month?j
 */


#ifndef GLPA_MACRO_CG_H_
#define GLPA_MACRO_CG_H_

#include <vector>

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