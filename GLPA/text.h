#ifndef TEXT_H_
#define TEXT_H_

#include <Windows.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <tchar.h>

#include "color.h"

#include "vector.cuh"

class Text
{
public :
    void createFont(HDC h_buffer_dc, int size, std::wstring name, Rgb color, BOOL bold);
    void releaseFont();
    void addText(std::wstring textName, std::wstring text);
    void addTextGroup(std::vector<std::wstring> text_group);
    void drawText(HDC h_buffer_dc, Vec2d text_position, std::wstring text_name);
    void drawTextGroup();

private :
    HFONT font;
    std::unordered_map<std::wstring, std::wstring> textData;
    std::unordered_map<std::wstring, std::vector<std::wstring>> groupTextData;

};

#endif  TEXT_H_
