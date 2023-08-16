
#include "load.h"

int FILELOAD::loadBinary()
{
    char filePath[] = "C:/Users/nari-/Documents/GitHub/GLPA/x64/Debug/blueimage.bmp";
    //�t�@�C��������o�C�i���t�@�C���œǂݍ���
    std::ifstream ifs(filePath, std::ios::binary);
    if (ifs.fail())
    {
        return 1;
    }

    //�Ǎ��T�C�Y�𒲂ׂ�B
    ifs.seekg(0, std::ios::end);
    long long int size = ifs.tellg();
    ifs.seekg(0);

    //�ǂݍ��񂾃f�[�^��char�^�ɏo�͂���
    char *data = new char[size];
    ifs.read(data, size);

    //�T�C�Y���o�͂���
    OutputDebugStringA(("size = " + std::to_string(size) + "\n").c_str());
    std::cout << "size = "<< size <<"\n" ;
    for (int i = 1; i < size + 1; i++)
    {
        //�o�͂���
        OutputDebugStringA((std::to_string(data[i - 1]) + " ").c_str());
        std::cout <<data[i - 1] << " ";
        //16�o�C�g���ɉ��s����
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