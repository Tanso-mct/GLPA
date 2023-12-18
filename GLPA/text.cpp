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


void Text::drawText(HDC hBufDC, std::wstring groupName, int startLine){
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
            if(drawLine(
                hBufDC,
                groupName,
                startLine, 
                lines,
                &drawLines,
                it
            )){
                break;
            }
            // if (
            //     lines >= startLine &&
            //     (data[groupName].posTopLeft.y + drawLines * data[groupName].textSize * GLPA_TEXT_LINE_RATIO) <= 
            //     (data[groupName].posBottomRight.y - data[groupName].textSize * GLPA_TEXT_LINE_RATIO * 2)
            // ){
            //     TextOut(
            //         hBufDC,
            //         data[groupName].posTopLeft.x,
            //         data[groupName].posTopLeft.y + drawLines * data[groupName].textSize * GLPA_TEXT_LINE_RATIO,
            //         it.c_str(),
            //         _tcslen(it.c_str())
            //     );
            //     drawLines += 1;
            // }
            // else if(
            //     (data[groupName].posTopLeft.y + drawLines * data[groupName].textSize * GLPA_TEXT_LINE_RATIO) > 
            //     (data[groupName].posBottomRight.y - data[groupName].textSize * GLPA_TEXT_LINE_RATIO * 2)
            // ){
            //     break;
            // }
            lines += 1;
        }
        else{
            while (true)
            {
                if(drawLine(
                    hBufDC,
                    groupName,
                    startLine, 
                    lines,
                    &drawLines,
                    it.substr(cutIndex, wordsPerWidth)
                )){
                    break;
                }
                
                // if (
                //     lines >= startLine &&
                //     (data[groupName].posTopLeft.y + drawLines * data[groupName].textSize * GLPA_TEXT_LINE_RATIO) <= 
                //     (data[groupName].posBottomRight.y - data[groupName].textSize * GLPA_TEXT_LINE_RATIO * 2)
                // ){
                //     TextOut(
                //         hBufDC,
                //         data[groupName].posTopLeft.x,
                //         data[groupName].posTopLeft.y + drawLines * data[groupName].textSize * GLPA_TEXT_LINE_RATIO,
                //         it.substr(cutIndex, wordsPerWidth).c_str(),
                //         _tcslen(it.substr(cutIndex, wordsPerWidth).c_str())
                //     );
                //     drawLines += 1;
                // }else if(
                //     (data[groupName].posTopLeft.y + drawLines * data[groupName].textSize * GLPA_TEXT_LINE_RATIO) > 
                //     (data[groupName].posBottomRight.y - data[groupName].textSize * GLPA_TEXT_LINE_RATIO * 2)
                // ){
                //     break;
                // }

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


void Text::releaseGroup(std::wstring groupName){
    DeleteObject(data[groupName].font);
    data.erase(groupName);
}

