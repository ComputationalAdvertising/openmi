#include <array>
#include <iostream>

struct Data {
  Data(int _i, int _j): i(_i), j(_j) {}
  Data(): i(0), j(0) {}
  void Print() {
    std::cout << "i:" << i << ", j:" << j << std::endl;
  }
  int i;
  int j;
};

int main(int argc, char** argv) {
  std::array<Data, 100> arr;
  Data d5(5, 50);
  arr[5] = d5;
  Data d10(10, 100);
  arr[10] = d10;
  arr[5].Print();
  arr[4].Print();
  arr[10].Print();
  arr[5] = d10;
  arr.at(5).Print();
  return 0;
}
