#ifndef TEXT_H_
#define TEXT_H_

#include <Windows.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <tchar.h>

#include "color.h"

#include "vector.cuh"

#define GLPA_TEXT_ASPECT 2.1
#define GLPA_TEXT_LINE_RATIO 1.2

#define GLPA_SYSTEM_FIXED_FONT L"SYSTEM_FIXED_FONT"

typedef struct tagTEXT_GROUP{
    HFONT font;
    int textSize;
    Vec2d posTopLeft;
    Vec2d posBottomRight;
    Rgb color;
    std::vector<std::wstring> text;
} TextGroup;

#define GLPA_TEXT_EDIT_GROUP_LAST 0

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

    void edit();

    void releaseGroup(std::wstring group_name);

    bool drawLine(HDC h_buffer_dc, std::wstring group_name, int start_line, int now_line, int* draw_lines, std::wstring line_text);

    void drawText(HDC h_buffer_dc, std::wstring group_name, int start_line);
private :
    std::unordered_map<std::wstring, HFONT> font;
    std::unordered_map<std::wstring, TextGroup> data;

};

#endif  TEXT_H_
