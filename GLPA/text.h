#ifndef TEXT_H_
#define TEXT_H_

#include <Windows.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <tchar.h>

#include "color.h"

#include "vector.cuh"

typedef struct tagTEXT_GROUP{
    HFONT font;
    int textSize;
    Vec2d posTopLeft;
    Vec2d posBottomRight;
    Rgb color;
    std::vector<std::wstring> text;
} TextGroup;

class Text
{
public :
    void addGroup(
        std::wstring group_name,
        int arg_size, 
        std::wstring arg_font_name,
        Rgb arg_rgb,
        BOOL arg_bold,
        Vec2d arg_pos_top_left,
        Vec2d arg_pot_bottom_right
    );

    void addText(std::wstring group_name, std::wstring text);

    void releaseGroup(std::wstring group_name);

    void drawText(HDC h_buffer_dc, std::wstring group_name);

private :
    std::unordered_map<std::wstring, HFONT> font;
    std::unordered_map<std::wstring, TextGroup> data;

};

#endif  TEXT_H_
