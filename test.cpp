#include <cstddef>
#include <iostream>

struct MyStruct {
    int    b; // 4 bytes
    double c;
    char   d;
    char   a;
};

int main() {
    std::cout << "Size of MyStruct: " << sizeof(MyStruct) << std::endl; // 输出的大小可能是24字节

    std::cout << "Offset of a: " << offsetof(MyStruct, a) << std::endl; // 输出的偏移量可能是0
    std::cout << "Offset of b: " << offsetof(MyStruct, b) << std::endl; // 输出的偏移量可能是4
    std::cout << "Offset of c: " << offsetof(MyStruct, c) << std::endl; // 输出的偏移量可能是8
    std::cout << "Offset of d: " << offsetof(MyStruct, d) << std::endl; // 输出的偏移量可能是16

    return 0;
}