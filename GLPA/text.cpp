#include "text.h"

void Text::createFont(int size, std::wstring name, BOOL bold){
    font = CreateFont(
		size, 0,				//高さ, 幅
		0, 0,					//角度1, 角度2
		bold ? FW_BOLD : FW_DONTCARE,	//太さ
		FALSE, FALSE, FALSE,	//斜体, 下線, 打消し線
		SHIFTJIS_CHARSET,		//文字セット
		OUT_DEFAULT_PRECIS,		//精度
		CLIP_DEFAULT_PRECIS,	//精度
		DEFAULT_QUALITY,		//品質
		DEFAULT_PITCH | FF_DONTCARE, //ピッチとファミリ
		name.c_str()
    );
}

void Text::addText(std::wstring textName, std::wstring text){
    textData.emplace(textName, text);
}

void Text::drawText(HDC hBufDC, Vec2d textPos, std::wstring textName){
    TextOut(
        hBufDC,
        textPos.x,
        textPos.y,
        textData[textName].c_str(),
        _tcslen(textData[textName].c_str())
    );
}
