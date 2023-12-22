#include "text.h"

void Text::addGroup(
    std::wstring groupName, 
    int argSize, 
    std::wstring argFontName, 
    Rgb argRgb, 
    BOOL argBold, 
    Vec2d argPosTopLeft,
    Vec2d argPosBottomRight,
    bool argVisible
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
    tempTextGroup.visible = argVisible;

    data.emplace(groupName, tempTextGroup);
}


void Text::addText(std::wstring groupName, std::wstring argText){
    data[groupName].text.push_back(argText);
}


std::wstring Text::getGroupOnMouse(LPARAM lParam, int dpi){
    for (auto it : data){
        if (
            LOWORD(lParam) * dpi >= it.second.posTopLeft.x &&
            LOWORD(lParam) * dpi <= it.second.posBottomRight.x &&

            HIWORD(lParam) * dpi >= it.second.posTopLeft.y &&
            HIWORD(lParam) * dpi <= it.second.posBottomRight.y
        ){
            return it.first;
        }
    }

    return GLPA_NULL_WTEXT;
}


std::wstring Text::getGroupLastLineWstr(std::wstring targetGroupName){
    return data[targetGroupName].text[
        data[targetGroupName].text.size() - 1
    ];
}

void Text::edit(std::wstring targetGroupName, int editType, std::wstring editText){
    if (editType == GLPA_TEXT_EDIT_GROUP_LAST){
        data[targetGroupName].text[
            data[targetGroupName].text.size() - 1
        ] = editText;
    }
    else{
        throw std::runtime_error(ERROR_TEXT_EDIT);
    }
}


bool Text::drawLine(HDC hBufDC, std::wstring groupName, int startLine, int nowLine, int* drawLines, std::wstring lineText){
    if (
        nowLine >= startLine &&
        (data[groupName].posTopLeft.y + *drawLines * data[groupName].textSize * GLPA_TEXT_LINE_RATIO) <= 
        (data[groupName].posBottomRight.y - data[groupName].textSize * GLPA_TEXT_LINE_RATIO * 2)
    ){
        TextOut(
            hBufDC,
            data[groupName].posTopLeft.x,
            data[groupName].posTopLeft.y + *drawLines * data[groupName].textSize * GLPA_TEXT_LINE_RATIO,
            lineText.c_str(),
            _tcslen(lineText.c_str())
        );
        *drawLines += 1;
    }
    else if(
        (data[groupName].posTopLeft.y + *drawLines * data[groupName].textSize * GLPA_TEXT_LINE_RATIO) > 
        (data[groupName].posBottomRight.y - data[groupName].textSize * GLPA_TEXT_LINE_RATIO * 2)
    ){
        return true;
    }

    return false;
}


void Text::drawText(HDC hBufDC, std::wstring groupName){
    if(data[groupName].visible){
        SetBkMode(hBufDC, TRANSPARENT);
        SetTextColor(hBufDC, RGB(data[groupName].color.r, data[groupName].color.g, data[groupName].color.b));

        SelectObject(hBufDC, data[groupName].font);

        double width = data[groupName].posBottomRight.x - data[groupName].posTopLeft.x;

        int wordsPerWidth = std::round(width / (data[groupName].textSize / GLPA_TEXT_ASPECT));

        int lines = 0;
        int drawLines = 0;

        int cutIndex = 0;

        for (auto it : data[groupName].text){
            if (it.size() <= wordsPerWidth){
                if(drawLine(hBufDC, groupName, data[groupName].startLine, lines, &drawLines, it)){
                        break;
                    }
                lines += 1;
            }
            else{
                while (true)
                {
                    if(drawLine(hBufDC, groupName, data[groupName].startLine, lines, &drawLines, it.substr(cutIndex, wordsPerWidth))){
                        break;
                    }
                    
                    lines += 1;
                    cutIndex += wordsPerWidth;

                    if (cutIndex >= it.size()){
                        cutIndex = 0;
                        break;
                    }
                }
                
            }
        }
    }
}


void Text::setStartLine(std::wstring groupName, int startLine){
    if(data.find(groupName) != data.end()){
        data[groupName].startLine = startLine;
    }
}


void Text::drawAll(HDC hBufDC){
    for (auto it : data){
        drawText(hBufDC, it.first);
    }
}


void Text::releaseGroup(std::wstring groupName){
    DeleteObject(data[groupName].font);
    data.erase(groupName);
}


void Text::releaseAllGroup(){
    for (auto& it : data){
        DeleteObject(data[it.first].font);
    }
}

