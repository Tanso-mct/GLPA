
#include "load.h"

int FILELOAD::loadBinary(std::string file_type, std::string file_name)
{
    string filePath ("../x64/Debug/sample.png");

    //ファイル名からバイナリファイルで読み込む
    std::ifstream ifs(filePath, std::ios::binary);
    if (ifs.fail())
    {
        OutputDebugStringA("file open failed\n");
        return 1;
    }

    //読込サイズを調べる。
    ifs.seekg(0, ios::end);
    long long int size = ifs.tellg();
    ifs.seekg(0);
    vector<string> imageData (size);

    //読み込んだデータをchar型に出力する
    char *fileData = new char[size];
    ifs.read(fileData, size);

    //サイズを出力する
    // OutputDebugStringA(("size = " + std::to_string(size) + "\n").c_str());

    char hexChar[9];
    int decimal;

    //バイナリデータの格納
    for (int i = 1; i < size + 1; i++)
    {
        decimal = stoi(to_string(fileData[i - 1]));
        // char hex_str[9];
        sprintf_s(hexChar, sizeof(hexChar),"%.2X", decimal);
        string hexString (hexChar);
        if (stoi(to_string(fileData[i - 1])) < 0)
        {
            hexString.erase(0, 6);
            imageData[i - 1] = hexString;
        }
        else
        {
            imageData[i - 1] = hexString;
        }
    }

    // OutputDebugStringA("\nEND\n");

    // //16進数バイナリデータを表示する
    // for (int i = 1; i < imageData.size() + 1; i++)
    // {
    //     OutputDebugStringA("   ");
    //     OutputDebugStringA((imageData[i - 1]).c_str());

    //     if ((i % 16) == 0)
    //     {
    //         OutputDebugStringA("\n");
    //     }
    // }

    delete fileData;
    return 0;
}

LOAD_BMP sampleBmpFile;