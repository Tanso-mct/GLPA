#ifndef TEXT_H_
#define TEXT_H_

#include <Windows.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <tchar.h>

#include "color.h"
#include "vector.cuh"
#include "error.h"

#define GLPA_TEXT_ASPECT 2.1
#define GLPA_TEXT_LINE_RATIO 1.1

#define GLPA_SYSTEM_FIXED_FONT L"SYSTEM_FIXED_FONT"

typedef struct tagTEXT_GROUP{
    HFONT font;
    int textSize;
    Vec2d posTopLeft;
    Vec2d posBottomRight;
    Rgb color;
    bool visible;
    std::vector<std::wstring> text;
    int startLine = 0;
} TextGroup;

#define GLPA_TEXT_EDIT_GROUP_LAST 0

#define GLPA_NULL_WTEXT L"NULL"

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
        Vec2d arg_pos_bottom_right,
        bool arg_visible
    );

    void addText(std::wstring group_name, std::wstring text);

    std::wstring getGroupOnMouse(LPARAM l_param, int dpi);

    std::wstring getGroupLastLineWstr(std::wstring target_group_name);

    void edit(std::wstring target_group_name, int edit_type, std::wstring edit_text);

    void releaseGroup(std::wstring group_name);

    bool drawLine(HDC h_buffer_dc, std::wstring group_name, int start_line, int now_line, int* draw_lines, std::wstring line_text);

    void drawText(HDC h_buffer_dc, std::wstring group_name);

    void setStartLine(std::wstring group_name, int start_line);

    void drawAll(HDC h_buffer_dc);

    void releaseAllGroup();
private :
    std::unordered_map<std::wstring, HFONT> font;
    std::unordered_map<std::wstring, TextGroup> data;

};

#endif  TEXT_H_
