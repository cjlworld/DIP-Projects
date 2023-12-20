#include <bits/stdc++.h>
#include <windows.h>

using namespace std;

/*
    24 位真彩 BMP
*/
struct BitMap
{
    /* 位图文件头:
    typedef struct tagBITMAPFILEHEADER {
        WORD    bfType; // 指定文件类型，必须是 0x4D42，即字符串“BM”，也就是说所有.bmp文件的头两个字节都是“BM”。
        DWORD   bfSize; // 指定文件大小，包括这14个字节。
        WORD    bfReserved1; // 为保留字，不用考虑, must be 0 
        WORD    bfReserved2; // 为保留字，不用考虑, must be 0
        DWORD   bfOffBits; // 为从文件头实际的位图数据的偏移字节数，即图1.3中前三个部分的长度之和。
    } BITMAPFILEHEADER;
    // WORD为无符号16位整数，DWORD为无符号32位整数
    */
    BITMAPFILEHEADER bitMapFileHeader;

    /* 位图信息头 BITMAPINFOHEADER
    typedef struct tagBITMAPINFOHEADER{
        DWORD  biSize; // 指定这个结构的长度，为40。
        LONG   biWidth; // 指定图象的宽度，单位是象素。
        LONG   biHeight; // 指定图象的高度，单位是象素。
        WORD   biPlanes; // 必须是1，不用考虑。
        WORD   biBitCount; // 指定表示颜色时要用到的位数，常用的值为1(黑白二色图), 4(16色图), 8(256色), 24(真彩色图)
        DWORD  biCompression; // 指定位图是否压缩
        DWORD  biSizeImage; // 指定实际的位图数据占用的字节数，其实也可以从以下的公式中计算出来：biSizeImage=biWidth’ × biHeight
        LONG   biXPelsPerMeter; // 指定目标设备的水平分辨率，单位是每米的象素个数，
        LONG   biYPelsPerMeter; // 指定目标设备的垂直分辨率
        DWORD  biClrUsed; // 指定本图象实际用到的颜色数，如果该值为零，则用到的颜色数为2^biBitCount。
        DWORD  biClrImportant; // 指定本图象中重要的颜色数，如果该值为零，则认为所有的颜色都是重要的。
    } BITMAPINFOHEADER;
    */
    BITMAPINFOHEADER bitMapInfoHeader;

    int width, height; 
    // width = bitMapInfoHeader.biWidth
    // height = bitMapInfoHeader.biHeight

    // bitMap 中的 RGB 是 Byte
    // 注意顺序，因为高位到低位是 BGR
    struct RGBBytes 
    {
        BYTE B, G, R;
        RGBBytes(BYTE B, BYTE G, BYTE R) :  B(B), G(G), R(R)  {}
        RGBBytes() {}
    };

    vector< vector<RGBBytes> > bitMap;

    // 读入 bmp 文件
    void readFile(const char *bitMapFilePath)
    {   
        // fprintf(stderr, "bitMapFilePath = %s\n", bitMapFilePath);
        // BITMAPFILEHEADER.bfType 的值必须为 0x4D42
        static const WORD BFTYPE_NUM = 0x4D42;

        FILE *fp = fopen(bitMapFilePath, "rb");

        // 首先读入 bitMapFileHeader
        // size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
        fread(&bitMapFileHeader, sizeof bitMapFileHeader, 1, fp);
        // 验证
        // fprintf(stderr, "0x%x\n", BFTYPE_NUM);
        // fprintf(stderr, "0x%x\n", bitMapFileHeader.bfType);
        assert(bitMapFileHeader.bfType == BFTYPE_NUM);
        assert(bitMapFileHeader.bfReserved1 == 0);
        assert(bitMapFileHeader.bfReserved2 == 0);

        // 读入 bitMapInfoHeader
        fread(&bitMapInfoHeader, sizeof bitMapInfoHeader, 1, fp);
        // 验证是否是 24 位真彩色图
        // fprintf(stderr, "bitMapInfoHeader.biBitCount = %d", (int)bitMapInfoHeader.biBitCount);
        assert(bitMapInfoHeader.biBitCount == 24);

        // 是否压缩
        assert(bitMapInfoHeader.biCompression == BI_RGB);

        // cerr << bitMapInfoHeader.biClrUsed << endl;
        // cerr << bitMapFileHeader.bfOffBits << endl;
        // 读入长宽
        width = bitMapInfoHeader.biWidth;
        height = bitMapInfoHeader.biHeight;

        // 真彩色图不需要调色板，直接读！
        // 创建一个 height * width 的 bitMap
        bitMap = vector< vector<RGBBytes> >(height, vector<RGBBytes>(width));

        fseek(fp, bitMapFileHeader.bfOffBits, SEEK_SET);
        // 读入
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                fread(&bitMap[i][j], sizeof(RGBBytes), 1, fp);
                // fprintf(stderr, "bitMap(%d, %d) = (R: %d, G: %d, B: %d)\n", 
                //         i, j, (int)bitMap[i][j].R, (int)bitMap[i][j].G, (int)bitMap[i][j].B);
            }
            BYTE blankByte;
            for(int j = 0; j < (4-3*width%4)%4; j++) 
                fread(&blankByte, sizeof(blankByte), 1, fp);
        }

        fclose(fp);
    }

    void writeFile(const char *bitMapFilePath) const
    {
        FILE *fp = fopen(bitMapFilePath, "wb");

        if(fp == NULL) {
            cerr << "error: BMP File not found!" << endl;
            exit(0);
        }

        // 输出文件头
        // size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)
        fwrite(&bitMapFileHeader, sizeof bitMapFileHeader, 1, fp);
        fwrite(&bitMapInfoHeader, sizeof bitMapInfoHeader, 1, fp);

        // 输出位图数据
        fseek(fp, bitMapFileHeader.bfOffBits, SEEK_SET);
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                fwrite(&bitMap[i][j], sizeof(RGBBytes), 1, fp);
            }
            // 每行的字节数必须是 4 的倍数，补齐
            BYTE blankByte = 0;
            for(int j = 0; j < (4-3*width%4)%4; j++) 
                fwrite(&blankByte, sizeof(BYTE), 1, fp);
        }

        fclose(fp);
    }
    
    // 清空位图
    void clear() 
    {
        height = width = 0;
        bitMap.clear();
    }

    // 读入并创建
    BitMap(const char *bitMapFilePath) { this->readFile(bitMapFilePath); }
};


struct YUVPixel;

// 实际的 R, G, B 是连续的, 因此使用 double
struct RGBPixel
{
    double R, G, B;

    RGBPixel(double R, double G, double B) : R(R), G(G), B(B) {}
    RGBPixel() {}

    // 将 R, G, B 规范到 [0, 255] 的区间
    void norm()
    {
        if(R <= 0) R = 0;
        if(R >= 255) R = 255;
        if(G <= 0) G = 0;
        if(G >= 255) G = 255;
        if(B <= 0) B = 0;
        if(B >= 255) B = 255;
    }

    YUVPixel toYUVPixel() const;
};

struct YUVPixel
{
    double Y, U, V;

    YUVPixel(double Y, double U, double V) : Y(Y), U(U), V(V) {}
    YUVPixel() {}

    // 将 Y 规范到 [0, 255] 的区间
    void normY()
    {
        if(Y <= 0) Y = 0;
        if(Y >= 255) Y = 255; 
    }

    RGBPixel toRGBPixel() const;
};

// RGB 转化到 YUV 颜色空间 
YUVPixel RGBPixel::toYUVPixel() const
{
    YUVPixel pixelYUV;
    pixelYUV.Y = 0.299*R + 0.587*G + 0.114*B;
    pixelYUV.U = 0.493*(B - pixelYUV.Y);
    pixelYUV.V = 0.877*(R - pixelYUV.Y);
    pixelYUV.normY(); // 调整亮度在 [0, 255]
    return pixelYUV;
}

// YUV 转换到 RGB 颜色空间
RGBPixel YUVPixel::toRGBPixel() const
{
    RGBPixel pixelRGB;
    // r = y                  + (1.370705 * v);
    // g = y - (0.337633 * u) - (0.698001 * v);
    // b = y + (1.732446 * u);
    pixelRGB.R = Y + (1.370705 * V);
    pixelRGB.G = Y - (0.337633 * U) - (0.698001 * V);
    pixelRGB.B = Y + (1.732446 * U);
    pixelRGB.norm();
    return pixelRGB;
}

struct YUVColorMap;

// R, G, B 颜色空间的像素图
struct RGBColorMap
{
    int width, height;
    vector< vector<RGBPixel> > bitMap;

    // 初始化
    RGBColorMap(int height = 0, int width = 0) 
    {
        this->width = width; this->height = height;
        bitMap.resize(height, vector<RGBPixel>(width));
    }

    // 从 BMP 中读入位图数据
    void readBMP(const BitMap &bmp) 
    {
        this->width = bmp.width; this->height = bmp.height;
        bitMap.resize(height, vector<RGBPixel>(width));
        
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                bitMap[i][j] = RGBPixel(bmp.bitMap[i][j].R, bmp.bitMap[i][j].G, bmp.bitMap[i][j].B);
            }
        }
    }

    // 输出位图数据到 BMP
    void writeBMP(BitMap &bmp) const
    {
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                bmp.bitMap[i][j] = BitMap::RGBBytes((BYTE)bitMap[i][j].B, (BYTE)bitMap[i][j].G, (BYTE)bitMap[i][j].R);
            }
        }
    }

    // 产生灰度图
    // 不改变原来的值，返回一个新的 colMap，为灰度图
    RGBColorMap toGrayscale() const 
    {
        RGBColorMap colMap(height, width);

        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                YUVPixel pixelYUV = bitMap[i][j].toYUVPixel();
                // toYUVPixel() 会自动调整亮度
                colMap.bitMap[i][j] = RGBPixel(pixelYUV.Y, pixelYUV.Y, pixelYUV.Y);
            }
        }

        return colMap;
    }

    YUVColorMap toYUVColorSpace() const;
};

// Y, U, V 颜色空间的像素图
struct YUVColorMap
{
    int width, height;
    vector< vector<YUVPixel> > bitMap;

    // 初始化
    YUVColorMap(int height = 0, int width = 0) 
    {
        this->width = width; this->height = height;
        bitMap.resize(height, vector<YUVPixel>(width));
    }

    // 加上一定亮度, 不改变原有的值，返回一个新的 colMap
    YUVColorMap addY(double deltaY) const
    {
        YUVColorMap colMap(height, width);
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                colMap.bitMap[i][j] = bitMap[i][j];
                colMap.bitMap[i][j].Y += deltaY;
                colMap.bitMap[i][j].normY();
            }
        }
        return colMap;
    }

    RGBColorMap toRGBColorSpace() const;
};

// 将 RGB 颜色空间的位图 变为 YUV 位图
YUVColorMap RGBColorMap::toYUVColorSpace() const
{
    YUVColorMap colMap(height, width);
    // cerr << height << " " << width << endl;
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            colMap.bitMap[i][j] = bitMap[i][j].toYUVPixel();
        }
    }
    return colMap;
}

// 将 YUV 位图变为 RGB 位图
RGBColorMap YUVColorMap::toRGBColorSpace() const
{
    RGBColorMap colMap(height, width);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            colMap.bitMap[i][j] = bitMap[i][j].toRGBPixel();
            colMap.bitMap[i][j].norm();
        }
    }
    return colMap;
}

int main()
{
    // 输入要处理的 bmp 地址
    string bmpFilePath;
    cout << "Please enter the path of the bmp file you want to convert (example: \"PIC-1.bmp\"): ";
    cin >> bmpFilePath;
    // 输入要增加的亮度
    double deltaY;
    cout << "Please enter the luminance to be changed: ";
    cin >> deltaY;

    // 新建 bmp, 并读入
    BitMap bmp(bmpFilePath.c_str());
    // 新建 RGB 位图，从 bmp 中读入数据
    RGBColorMap colMap; colMap.readBMP(bmp);
    // 输出灰度图
    colMap.toGrayscale().writeBMP(bmp);
    bmp.writeFile("grayscale.bmp");
    // 转换到 YUV, 增加 Y, 转换回 RGB, 输出图片
    colMap.toYUVColorSpace().addY(deltaY).toRGBColorSpace().writeBMP(bmp);
    bmp.writeFile("new_luminance.bmp");

    // 图片输出完成
    cout << "OK, The greyscale image has been output to grayscale.bmp and the new luminance image has been output to new_luminance.bmp." << endl;
    return 0;
}
