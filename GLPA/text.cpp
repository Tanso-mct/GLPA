#include "text.h"

void Text::addGroup(
    std::wstring groupName, 
    int argSize, 
    std::wstring argFontName, 
    Rgb argRgb, 
    BOOL argBold, 
    Vec2d argPosTopLeft,
    Vec2d argPosBottomRight
){
    TextGroup tempTextGroup;
    tempTextGroup.font = CreateFont(
		argSize, 0,
		0, 0,
		argBold ? FW_BOLD : FW_DONTCARE,
		FALSE, FALSE, FALSE,
		SHIFTJIS_CHARSET,
		OUT_DEFAULT_PRECIS,
		CLIP_DEFAULT_PRECIS,
		DEFAULT_QUALITY,
		DEFAULT_PITCH | FF_DONTCARE,
		argFontName.c_str()
    );

    tempTextGroup.textSize = argSize;

    tempTextGroup.color = argRgb;
    tempTextGroup.posTopLeft = argPosTopLeft;
    tempTextGroup.posBottomRight = argPosBottomRight;

    data.emplace(groupName, tempTextGroup);
}


void Text::addText(std::wstring groupName, std::wstring argText){
    data[groupName].text.push_back(argText);
}


void Text::drawText(HDC hBufDC, std::wstring groupName){
    SetBkMode(hBufDC, TRANSPARENT);
    SetTextColor(hBufDC, RGB(data[groupName].color.r, data[groupName].color.g, data[groupName].color.b));

    SelectObject(hBufDC, data[groupName].font);

    TextOut(
        hBufDC,
        data[groupName].posTopLeft.x,
        data[groupName].posTopLeft.y,
        data[groupName].text[0].c_str(),
        _tcslen(data[groupName].text[0].c_str())
    );
}


void Text::releaseGroup(std::wstring groupName){
    DeleteObject(data[groupName].font);
    data.erase(groupName);
}

