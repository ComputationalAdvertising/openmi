#include <iostream>

class Base {
public:
  virtual void Init(int i) { this->i = i; }

  virtual void Print() {
    std::cout << "Base::Print i: " << i << std::endl;
  }

protected:
  int i;
};

class SubBase : public Base {
public:
  void Init(int i) override {
    this->i = i;
  }

  void Print() override {
    std::cout << "SubBase::Print i: " << i << std::endl;
  }
};

class D : public SubBase {
public:
  void Print() override {
    std::cout << "D::Print i: " << i << std::endl;
  }
};

int main(int argc, char** argv) {
  SubBase s;
  s.Init(10);
  s.Print();

  D d;
  d.Init(100);
  d.Print();
  return 0;
}
