#ifndef CGMATH_H_
#define CGMATH_H_

#include <vector>

#define COLUMN1ST 0
#define COLUMN2ST 1
#define COLUMN3ST 2
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

class MATRIX
{
public :
    VECTOR3D sourceMatrix; // Value of 3D coordinates before calculation
    VECTOR3D calcuMatrix;  // Calculated 3D Coordinate 

    // Enter the value of the 3D coordinates before calculation
    void inputSourceMatrix(VECTOR3D input_num);

    // Calculation of each matrix
    void posTrans
    (
        VECTOR3D source_3d_coordinates,
        VECTOR3D change_pos_amount
    );
    void rotTrans
    (
        VECTOR3D source_3d_coordinates,
        int rotation_axis,
        double rotation_angle
    );
    void scaleTrans
    (
        VECTOR3D source_3d_coordinates,
        VECTOR3D scaling_rate
    );
};

#endif CGMATH_H_
