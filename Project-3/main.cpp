#include <map>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <windows.h>

using std::string;
using std::vector;
using std::map;
using std::cerr, std::endl, std::cin, std::cout;

/*
    24 位真彩 BMP
*/
class BitMap
{
public:
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

    std::vector< std::vector<RGBBytes> > bitMap;

    // 读入 bmp 文件
    void readFile(string bitMapFilePath)
    {   
        // fprintf(stderr, "bitMapFilePath = %s\n", bitMapFilePath);
        // BITMAPFILEHEADER.bfType 的值必须为 0x4D42
        static const WORD BFTYPE_NUM = 0x4D42;

        FILE *fp = fopen(bitMapFilePath.c_str(), "rb");

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
        bitMap.clear();
        bitMap.resize(height, vector<RGBBytes>(width));

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

    void writeFile(string bitMapFilePath) const
    {
        FILE *fp = fopen(bitMapFilePath.c_str(), "wb");

        if(fp == NULL) {
            std::cerr << "error: BMP File not found!" << std::endl;
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
    BitMap(string bitMapFilePath) { this->readFile(bitMapFilePath); }
    // 创建空的 BMP
    BitMap() {}
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
    // Y 为 亮度

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

template<class T>
class ColorMap
{
private:

public:
    int width, height;
    std::vector< std::vector<T> > bitMap;

    // 初始化
    ColorMap(int height = 0, int width = 0)
    {
        this->width = width; this->height = height;
        bitMap.clear();
        bitMap.resize(height, std::vector<T>(width));
    }
};

class YUVColorMap;
class RGBColorMap;
class GrayscaleColorMap;
class BinaryColorMap;

// 灰度图
class GrayscaleColorMap: public ColorMap<double>
{
private:
public:
    // 构造
    GrayscaleColorMap(int height = 0, int width = 0): ColorMap<double>(height, width) {}  
    // 转换到 RGB 空间
    RGBColorMap toRGBColorSpace() const;
    // 转换到 二值 空间
    BinaryColorMap toBinaryColorSpace(int blockH, int blockW, double alpha) const;
};

// 结构元素
typedef std::vector< std::pair<int, int> > StructureElement;

// 十字结构元素
const StructureElement CROSS_STRUCTURE_ELEMENT = StructureElement{
    std::pair<int, int>(0, 0),
    std::pair<int, int>(1, 0),
    std::pair<int, int>(-1, 0),
    std::pair<int, int>(0, 1),
    std::pair<int, int>(0, -1)
};

// 二值图像
// 白色为 0, 黑色为 1
class BinaryColorMap: public ColorMap<unsigned int>
{
public:
    // 构造
    BinaryColorMap(int height = 0, int width = 0): ColorMap<unsigned>(height, width) {}

    // 转化到 灰度图
    GrayscaleColorMap toGrayscaleColorSpace() const;

    // 腐蚀
    BinaryColorMap erosion(StructureElement elements) const;
    // 膨胀
    BinaryColorMap dilation(StructureElement elements) const;
    // 开运算
    BinaryColorMap opening(StructureElement e1, StructureElement e2) const;
    // 闭运算
    BinaryColorMap closing(StructureElement e1, StructureElement e2) const;
};

// R, G, B 颜色空间的像素图
class RGBColorMap: public ColorMap<RGBPixel>
{
private:

public:
    // 构造
    RGBColorMap(int height = 0, int width = 0): ColorMap<RGBPixel>(height, width) {}

    // 从 BMP 中读入位图数据
    void readBMP(const BitMap &bmp) 
    {
        this->width = bmp.width; this->height = bmp.height;
        bitMap.clear();
        bitMap.resize(height, std::vector<RGBPixel>(width));
        
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
    GrayscaleColorMap toGrayscaleColorSpace() const;

    // 转化为 YUV 图
    YUVColorMap toYUVColorSpace() const;

    // 直方图均衡化
    RGBColorMap HistogramEqualizeRGB() const;

    // RGB 对数操作
    RGBColorMap logarithmicOperationRGB() const;
};

// Y, U, V 颜色空间的像素图
class YUVColorMap: public ColorMap<YUVPixel>
{
public:
    // 构造
    YUVColorMap(int height = 0, int width = 0): ColorMap<YUVPixel>(height, width) {}

    // 转化为 RGB 图
    RGBColorMap toRGBColorSpace() const;
    // 亮度对数操作
    YUVColorMap logarithmicOperationY() const;
    // 直方图均衡化 Y
    YUVColorMap HistogramEqualizeY() const;
};

// 将 RGB 图变为灰度图
GrayscaleColorMap RGBColorMap::toGrayscaleColorSpace() const 
{
    GrayscaleColorMap colMap(height, width);

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            YUVPixel pixelYUV = bitMap[i][j].toYUVPixel();
            // toYUVPixel() 会自动调整亮度
            colMap.bitMap[i][j] = pixelYUV.Y;
        }
    }

    return colMap;
}

// 将 灰度图 变为 RGB 图
RGBColorMap GrayscaleColorMap::toRGBColorSpace() const
{
    RGBColorMap colMap(height, width);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            colMap.bitMap[i][j] = RGBPixel(bitMap[i][j], bitMap[i][j], bitMap[i][j]);
        }
    }
    return colMap;
}


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

// 将 二值图 变为 灰度图
GrayscaleColorMap BinaryColorMap::toGrayscaleColorSpace() const
{
    GrayscaleColorMap colMap(height, width);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            if(bitMap[i][j] == 0) {
                colMap.bitMap[i][j] = 255;
            } else {
                colMap.bitMap[i][j] = 0;
            }
        }
    }
    return colMap;
}

// 灰度图转换到二值空间
// blockH, blockW 表示处理的块的大小, -1 表示全局
BinaryColorMap GrayscaleColorMap::toBinaryColorSpace(int blockH = -1, int blockW = -1, double alpha = 1) const
{
    BinaryColorMap colMap(height, width);

    // 处理一个块
    static const auto transformBlock = [&](int u, int d, int l, int r) {
        static const int MAXN = 256;
        static int cnt[MAXN];
        static double sum[MAXN];

        // 计算 Miu between 结果范围 [0, 0.25]
        // maximize wb*wf*(uf-ub)^2
        // 结果越大越好
        static const auto calcMiuBetween = [&](int threshold) {
            assert(0 <= threshold && threshold <= 255);
            double N = (double)(d-u+1)*(double)(r-l+1);
            // f 为黑色, s 为白色
            double Nf = (threshold <= 0) ? 0 : cnt[threshold-1];
            double Nb = N-Nf;
            double wf = Nf/N*alpha;
            double wb = 1-wf;
            double Sf = (threshold <= 0) ? 0 : sum[threshold];
            double Sb = sum[MAXN-1]-Sf;
            double uf = Sf/Nf;
            double ub = Sb/Nb;

            double res = wb*wf*(uf-ub)*(uf-ub);
            return res;
        };

        // 预处理前缀和
        std::fill_n(cnt, MAXN, 0);
        std::fill_n(sum, MAXN, 0);

        for(int i = u; i <= d; i++) {
            for(int j = l; j <= r; j++) {
                double val = bitMap[i][j];
                assert(0 <= (unsigned)val && (unsigned)val <= 255);
                cnt[(unsigned)val]++;
                sum[(unsigned)val] += val;
            } 
        }

        for(int i = 1; i < MAXN; i++) {
            cnt[i] += cnt[i-1];
            sum[i] += sum[i-1];
        }

        double threshold = 0;
        double maxMiu = 0;
        for(int i = 0; i <= 255; i++) {
            double res = calcMiuBetween(i);
            if(res > maxMiu) {
                maxMiu = res;
                threshold = i;
            }
        }

        for(int i = u; i <= d; i++) {
            for(int j = l; j <= r; j++) {
                colMap.bitMap[i][j] = (bitMap[i][j] < threshold);
            }
        }
    };

    if(blockH <= 0) blockH = height;
    if(blockW <= 0) blockW = width;

    // 分块, 并处理不同的块
    for(int i = 0; i < height; i += blockH) {
        // 如果后一块不足, 按大的来
        int u = i, d = (i+2*blockH-1 >= height) ? height-1 : i+blockH-1;
        for(int j = 0; j < width; j += blockW) {
            // 如果后一块不足, 按大的来
            int l = j, r = (j+2*blockW-1 >= width) ? width-1 : j+blockW-1;
            transformBlock(u, d, l, r);
        }
    }

    return colMap;
}

// 二值图像腐蚀
// 默认结构元素为十字，也可以代入其他元素。
BinaryColorMap BinaryColorMap::erosion(const StructureElement e=CROSS_STRUCTURE_ELEMENT) const
{
    // cerr << "erosion" << endl;
    // 检查是否全部元素都是 1
    assert(e == CROSS_STRUCTURE_ELEMENT);
    static const auto checkIfAll = [this, e](int x, int y) -> unsigned {
        for(auto [dx, dy] : e) {
            int nx = x+dx, ny = y+dy;
            if(nx < 0 || nx >= height || ny < 0 || ny >= width) 
                continue;
            if(!bitMap[nx][ny]) return 0;
        }
        return 1;
    };

    BinaryColorMap colMap(height, width);

    for(int i = 0; i < height; i++) {   
        for(int j = 0; j < width; j++) {
            colMap.bitMap[i][j] = checkIfAll(i, j);
        }
    }

    return colMap;
}

// 二值图像膨胀
// 默认结构元素为十字，也可以代入其他元素。
BinaryColorMap BinaryColorMap::dilation(StructureElement e=CROSS_STRUCTURE_ELEMENT) const
{
    // cerr << "dilation" << endl;
    // 检查是否存在一个元素为 1
    static const auto checkIfOne = [this, e](int x, int y) -> unsigned {
        for(auto [dx, dy] : e) {
            // cerr << dx << " " << dy << endl;
            int nx = x+dx, ny = y+dy;
            if(nx < 0 || nx >= height || ny < 0 || ny >= width) 
                continue;
            if(bitMap[nx][ny]) return 1;
        }
        return 0;
    };

    BinaryColorMap colMap(height, width);

    for(int i = 0; i < height; i++) {   
        for(int j = 0; j < width; j++) {
            colMap.bitMap[i][j] = checkIfOne(i, j);
        }
    }

    return colMap;
}

// 二值图像开运算
// 默认结构元素为十字，也可以代入其他元素。
BinaryColorMap BinaryColorMap::opening(StructureElement e1=CROSS_STRUCTURE_ELEMENT, 
    StructureElement e2=CROSS_STRUCTURE_ELEMENT) const
{
    return this->erosion(e1).dilation(e2);
}

// 二值图像闭运算
// 默认结构元素为十字，也可以代入其他元素。
BinaryColorMap BinaryColorMap::closing(StructureElement e1=CROSS_STRUCTURE_ELEMENT, 
    StructureElement e2=CROSS_STRUCTURE_ELEMENT) const
{
    return this->dilation(e1).erosion(e2);
}

// Y 对数操作
YUVColorMap YUVColorMap::logarithmicOperationY() const 
{
    YUVColorMap colMap(height, width);
    double Lmax = 0;
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            Lmax = std::max(Lmax, bitMap[i][j].Y);
        }
    }
    double logLmax_1 = std::log(Lmax+1);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            colMap.bitMap[i][j] = bitMap[i][j];
            colMap.bitMap[i][j].Y = log(bitMap[i][j].Y+1)/logLmax_1*255;
        }
    }
    return colMap;
}

// RGB 对数操作
RGBColorMap RGBColorMap::logarithmicOperationRGB() const 
{
    RGBColorMap colMap(height, width);
    double Rmax = 0, Gmax = 0, Bmax = 0;
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            Rmax = std::max(Rmax, bitMap[i][j].R);
            Gmax = std::max(Gmax, bitMap[i][j].G);
            Bmax = std::max(Bmax, bitMap[i][j].B);
        }
    }
    double logRmax_1 = std::log(Rmax+1);
    double logGmax_1 = std::log(Gmax+1);
    double logBmax_1 = std::log(Bmax+1);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            colMap.bitMap[i][j] = RGBPixel(
                log(bitMap[i][j].R+1)/logRmax_1*255,
                log(bitMap[i][j].G+1)/logGmax_1*255,
                log(bitMap[i][j].B+1)/logBmax_1*255
            );
        }
    }
    return colMap;
}

// 直方图
// 采取 排序 的形式进行 积分 操作，是等价的
// 默认值类型为 double
// 默认值域为 [0, 255]
// 默认以 1e-4 为一阶
class Histogram
{
public:
    struct HistogramPoint {
        double val; // val 
        int x, y; // position

        friend bool operator<(const HistogramPoint &a, const HistogramPoint &b) {
            return a.val < b.val;
        }
    };

    vector<HistogramPoint> points;

    // 构造函数
    Histogram(const Histogram &other): points(other.points) {}
    Histogram() {}

    // 由 colMap 生成直方图
    template<class PixelType>
    void readColMap(const ColorMap<PixelType> &colMap, double getValue(const PixelType &pixel)) 
    {
        points.clear();
        for(int i = 0; i < colMap.height; i++) {
            for(int j = 0; j < colMap.width; j++) {
                points.push_back(HistogramPoint{
                    getValue(colMap.bitMap[i][j]),
                    i, j
                });
            }
        }
    }

    // 将经过变换的直方图赋给 colMap
    template<class PixelType>
    void writeColMap(ColorMap<PixelType> &colMap, void setValue(PixelType &pixel, double val)) 
    {   
        for(auto [val, x, y] : points) {
            setValue(colMap.bitMap[x][y], val);
        }
    }

    static constexpr double eps = 1e-4;
    
    // 进行直方图均衡化
    // 返回均衡化过后的新直方图
    Histogram Equalize() const 
    {
        Histogram his(*this);
        // 从小到大排序
        std::sort(his.points.begin(), his.points.end());
        for(int i = 0, j = 0; i < (int)his.points.size(); i = j+1) {
            // 找到同一色阶
            while(j+1 < his.points.size() && his.points[j+1].val < his.points[i].val+eps) 
                j++;
            // 对同一色阶赋色
            for(int k = i; k <= j; k++)
                his.points[k].val = (j+1)/((double)his.points.size())*255;
        }
        return his;
    }
};

// 直方图均衡化 Y
YUVColorMap YUVColorMap::HistogramEqualizeY() const
{
    YUVColorMap colMap(*this);
    Histogram his; 
    his.readColMap<YUVPixel>(colMap, [](const YUVPixel &pixel) { 
        return pixel.Y; 
    });
    his.Equalize().writeColMap<YUVPixel>(colMap, [](YUVPixel &pixel, double val) {
        pixel.Y = val;
    });
    return colMap;
}

// 直方图均衡化 R, G, B
RGBColorMap RGBColorMap::HistogramEqualizeRGB() const
{
    RGBColorMap colMap(*this);
    // 对 R, G, B 各做一次
    Histogram his; 
    // R
    his.readColMap<RGBPixel>(colMap, [](const RGBPixel &pixel) { 
        return pixel.R; 
    });
    his.Equalize().writeColMap<RGBPixel>(colMap, [](RGBPixel &pixel, double val) {
        pixel.R = val;
    });
    // G
    his.readColMap<RGBPixel>(colMap, [](const RGBPixel &pixel) { 
        return pixel.G; 
    });
    his.Equalize().writeColMap<RGBPixel>(colMap, [](RGBPixel &pixel, double val) {
        pixel.G = val;
    });
    // B
    his.readColMap<RGBPixel>(colMap, [](const RGBPixel &pixel) { 
        return pixel.B; 
    });
    his.Equalize().writeColMap<RGBPixel>(colMap, [](RGBPixel &pixel, double val) {
        pixel.B = val;
    });
    return colMap;
}

int main(int argc, char *args[])
{
    BitMap bmp;
    RGBColorMap colMap;
    BinaryColorMap binaryMap;

    string inStr(args[1]), outStr(args[2]), option(args[3]);
    cout << inStr << " " << outStr << endl;
    bmp.readFile(inStr);
    colMap.readBMP(bmp);
    if(option == "--logY") {
        colMap.toYUVColorSpace().logarithmicOperationY().toRGBColorSpace().writeBMP(bmp);
    } else if(option == "--logRGB") {
        colMap.logarithmicOperationRGB().writeBMP(bmp);
    } else if(option == "--eqY") {
        colMap.toYUVColorSpace().HistogramEqualizeY().toRGBColorSpace().writeBMP(bmp);
    } else if(option == "--eqRGB") {
        colMap.HistogramEqualizeRGB().writeBMP(bmp);
    } else {
        cerr << "error: NO OPTION!" << endl;
        assert(0);
    }
    bmp.writeFile(outStr);
    cout << "finish" << endl;
    return 0;
}
