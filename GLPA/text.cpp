#include "text.h"

void Text::createFont(HDC hBufDC, int size, std::wstring name, Rgb color, BOOL bold){
    font = CreateFont(
		size, 0,				//����, ��
		0, 0,					//�p�x1, �p�x2
		bold ? FW_BOLD : FW_DONTCARE,	//����
		FALSE, FALSE, FALSE,	//�Α�, ����, �ŏ�����
		SHIFTJIS_CHARSET,		//�����Z�b�g
		OUT_DEFAULT_PRECIS,		//���x
		CLIP_DEFAULT_PRECIS,	//���x
		DEFAULT_QUALITY,		//�i��
		DEFAULT_PITCH | FF_DONTCARE, //�s�b�`�ƃt�@�~��
		name.c_str()
    );

    SetTextColor(hBufDC, RGB(color.r, color.g, color.b));

    SelectObject(hBufDC, font);
}


void Text::addText(std::wstring textName, std::wstring text){
    textData.emplace(textName, text);
}


void Text::drawText(HDC hBufDC, Vec2d textPos, std::wstring textName){
    SetBkMode(hBufDC, TRANSPARENT);

    TextOut(
        hBufDC,
        textPos.x,
        textPos.y,
        textData[textName].c_str(),
        _tcslen(textData[textName].c_str())
    );
}


void Text::releaseFont(){
    DeleteObject(font);
}

