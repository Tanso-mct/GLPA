
#include "load.h"

int FILELOAD::loadBinary()
{
    char filePath[] = "C:/Users/nari-/Documents/GitHub/GLPA/x64/Debug/blueimage.bmp";
    //ファイル名からバイナリファイルで読み込む
    std::ifstream ifs(filePath, std::ios::binary);
    if (ifs.fail())
    {
        return 1;
    }

    //読込サイズを調べる。
    ifs.seekg(0, std::ios::end);
    long long int size = ifs.tellg();
    ifs.seekg(0);

    //読み込んだデータをchar型に出力する
    char *data = new char[size];
    ifs.read(data, size);

    //サイズを出力する
    OutputDebugStringA(("size = " + std::to_string(size) + "\n").c_str());
    std::cout << "size = "<< size <<"\n" ;
    for (int i = 1; i < size + 1; i++)
    {
        //出力する
        OutputDebugStringA((std::to_string(data[i - 1]) + " ").c_str());
        std::cout <<data[i - 1] << " ";
        //16バイト毎に改行する
        if ((i % 16) == 0)
        {
            OutputDebugStringA("\n");
            std::cout << "\n";
        }
    }
    std::cout << "\nEnd!\n"; 
    delete data;
    return 0;
}

LOAD_BMP sampleBmpFile;