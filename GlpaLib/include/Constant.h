#ifndef GLPA_CONSTANT_H_
#define GLPA_CONSTANT_H_

#include <string>

#include "ErrorHandler.h"

#define GPU_BOOL int

#define GPU_CO(condition, trueValue, falseValue) \
    condition ? trueValue : falseValue;

#define GPU_IF(condition, branch) \
    for(int branch = 0; branch < (condition) ? TRUE : FALSE; branch++) \

#define GPU_IS_EMPTY -1
#define GPU_IS_TRIANGLE 0
#define GPU_IS_SQUARE 1

#define GPU_MAX_FACE_V 4

#define GPU_VV_FACE_AMOUNT 6
#define GPU_VV_LINE_AMOUNT 12

#define GPU_POLY_LINE_AMOUNT 3

namespace Glpa 
{

constexpr int X = 0;
constexpr int Y = 1;
constexpr int Z = 2;

constexpr int SIZE_VEC2D = 2;
constexpr int SIZE_VEC3D = 3;

constexpr const char* CHAR_ENTER = "enter";
constexpr const char* CHAR_SPACE = "space";
constexpr const char* CHAR_ESCAPE = "escape";
constexpr const char* CHAR_TAB = "tab";
constexpr const char* CHAR_BACKSPACE = "back_space";

constexpr const char* CHAR_LSHIFT = "left_shift";
constexpr const char* CHAR_LCTRL = "left_ctrl";
constexpr const char* CHAR_LALT = "left_alt";
constexpr const char* CHAR_WIN = "window";

constexpr const char* CHAR_RSHIFT = "right_shift";
constexpr const char* CHAR_RCTRL = "right_ctrl";
constexpr const char* CHAR_RALT = "right_alt";

constexpr const char* CHAR_F1 = "f1";
constexpr const char* CHAR_F2 = "f2";
constexpr const char* CHAR_F3 = "f3";
constexpr const char* CHAR_F4 = "f4";
constexpr const char* CHAR_F5 = "f5";
constexpr const char* CHAR_F6 = "f6";
constexpr const char* CHAR_F7 = "f7";
constexpr const char* CHAR_F8 = "f8";
constexpr const char* CHAR_F9 = "f9";
constexpr const char* CHAR_F10 = "f10";
constexpr const char* CHAR_F11 = "f11";
constexpr const char* CHAR_F12 = "f12";

constexpr const char* CHAR_MOUSE_MOVE = "mouse_move";

constexpr const char* CHAR_MOUSE_RBTN_DOWN = "mouse_r_button_down";
constexpr const char* CHAR_MOUSE_RBTN_UP = "mouse_r_button_up";
constexpr const char* CHAR_MOUSE_RBTN_DBCLICK = "mouse_r_double_click";

constexpr const char* CHAR_MOUSE_LBTN_DOWN = "mouse_l_button_down";
constexpr const char* CHAR_MOUSE_LBTN_UP = "mouse_l_button_up";
constexpr const char* CHAR_MOUSE_LBTN_DBCLICK = "mouse_l_double_click";

constexpr const char* CHAR_MOUSE_MBTN_DOWN = "mouse_m_button_down";
constexpr const char* CHAR_MOUSE_MBTN_UP = "mouse_m_button_up";

constexpr const char* CHAR_MOUSE_WHEEL = "mouse_wheel";
constexpr const int INT_MOUSE_WHEEL_MOVE = 120;

constexpr const char* CLASS_SCENE_OBJECT = "class_scene_object";
constexpr const char* CLASS_FILE = "class_file";
constexpr const char* CLASS_PNG = "class_png";
constexpr const char* CLASS_IMAGE = "class_image";
constexpr const char* CLASS_TEXT = "class_txt";
constexpr const char* CLASS_STATIONARY_OBJECT = "class_stationary_object";
constexpr const char* CLASS_CAMERA = "class_camera";

constexpr const char* COLOR_BLACK = "color_black";
constexpr const char* COLOR_GREEN = "color_green";

constexpr const char* MATERIAL_DIFFUSE = "material_diffuse";
constexpr const char* MATERIAL_ORM = "material_occlusion_roughness_metallic";
constexpr const char* MATERIAL_NORMAL = "material_normal";

constexpr const float PI = 3.14159265358979323846f;

const enum VV_XZ_FACE
{
    NEAR_LEFT,
    FAR_LEFT,
    FAR_RIGHT,
    NEAR_RIGHT
};

const enum VV_YZ_FACE
{
    NEAR_TOP,
    FAR_TOP,
    FAR_BOTTOM,
    NEAR_BOTTOM
};

const enum FACE_3D
{
    TOP,
    FRONT,
    RIGHT,
    LEFT,
    BACK,
    BOTTOM
};

const enum RECT_3D
{
    FRONT_TOP_LEFT,
    FRONT_TOP_RIGHT,
    FRONT_BOTTOM_RIGHT,
    FRONT_BOTTOM_LEFT,
    BACK_TOP_LEFT,
    BACK_TOP_RIGHT,
    BACK_BOTTOM_RIGHT,
    BACK_BOTTOM_LEFT
};

const enum RECT_3D_LINE
{
    L1_START = Glpa::RECT_3D::FRONT_TOP_LEFT,
    L1_END = Glpa::RECT_3D::FRONT_TOP_RIGHT,

    L2_START = Glpa::RECT_3D::FRONT_TOP_RIGHT,
    L2_END = Glpa::RECT_3D::FRONT_BOTTOM_RIGHT,

    L3_START = Glpa::RECT_3D::FRONT_BOTTOM_RIGHT,
    L3_END = Glpa::RECT_3D::FRONT_BOTTOM_LEFT,

    L4_START = Glpa::RECT_3D::FRONT_BOTTOM_LEFT,
    L4_END = Glpa::RECT_3D::FRONT_TOP_LEFT,

    L5_START = Glpa::RECT_3D::FRONT_TOP_LEFT,
    L5_END = Glpa::RECT_3D::BACK_TOP_LEFT,

    L6_START = Glpa::RECT_3D::FRONT_TOP_RIGHT,
    L6_END = Glpa::RECT_3D::BACK_TOP_RIGHT,

    L7_START = Glpa::RECT_3D::FRONT_BOTTOM_RIGHT,
    L7_END = Glpa::RECT_3D::BACK_BOTTOM_RIGHT,

    L8_START = Glpa::RECT_3D::FRONT_BOTTOM_LEFT,
    L8_END = Glpa::RECT_3D::BACK_BOTTOM_LEFT,

    L9_START = Glpa::RECT_3D::BACK_TOP_LEFT,
    L9_END = Glpa::RECT_3D::BACK_TOP_RIGHT,

    L10_START = Glpa::RECT_3D::BACK_TOP_RIGHT,
    L10_END = Glpa::RECT_3D::BACK_BOTTOM_RIGHT,

    L11_START = Glpa::RECT_3D::BACK_BOTTOM_RIGHT,
    L11_END = Glpa::RECT_3D::BACK_BOTTOM_LEFT,

    L12_START = Glpa::RECT_3D::BACK_BOTTOM_LEFT,
    L12_END = Glpa::RECT_3D::BACK_TOP_LEFT
};

}

#endif GLPA_CONSTANT_H_