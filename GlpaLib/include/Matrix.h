#ifndef GLPA_MATRIX_H_
#define GLPA_MATRIX_H_

#include <initializer_list>

namespace Glpa
{
    
class Matrix
{
public :
    float m3x3[3][3] = {0};
    float m4x4[4][4] = {0};

    Matrix(std::initializer_list<float> list);
    ~Matrix();

    void set(std::initializer_list<float> list);

    void put(float matrix[3][3]);
    void put(float matrix[4][4]);
};

} // namespace Glpa


#endif GLPA_MATRIX_H_

