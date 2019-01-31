//#include "variable.h"
#include <iostream>
#include "node.h"

using namespace openmi;

void print_inputs(Node& node) {
  std::vector<Node*> inputs = node.Inputs();
  LOG(INFO) << "print_inputs name: " << node.Name() << ", its inputs: " << inputs.size();
  std::string link("node: " + node.Name());
  link += ", its inputs: ";
  if (inputs.size() == 0) {
    return;
  }
  for (int i = 0; i < inputs.size(); ++i) {
    print_inputs(*inputs[i]);
    link += " " + inputs[i]->Name();
  }
  LOG(INFO) << link;
}

int main(int argc, char** argv) {
  /*
  std::string v1_name("v1_name");
  std::string v2_name("v2_name");

  Variable* v1 = new Variable(v1_name);
  Variable* v2 = new Variable(v2_name);

  std::cout << "v1: " << v1->GetNode()->DebugString() << std::endl;
  std::cout << "v2: " << v2->GetNode()->DebugString() << std::endl;
  */

  // node +operator
  std::string n1_name("n1");
  Node n1(n1_name, 3);
  std::string n2_name("n2");
  Node n2(n2_name, 1);

  Node n3 = n1 + n2;
  std::string n3_name("n3");
  n3.SetName(n3_name);
  LOG(INFO) << " ================== NX ================";
  //Node nx = n2 + n3 + n1;
  Node nx = n2.Add(n3).Add(n1);
  std::string nx_name("nx");
  //nx.SetName(nx_name);
  std::cout << "nx.Inputs()[1]: " << nx.Inputs()[1]->DebugString() << std::endl;
  std::cout << "nx.Inputs()[0]: " << nx.Inputs()[0]->DebugString() << std::endl;
  std::cout << "n3: " << n3.DebugString() << std::endl;

  //Node n4 = n3 + nx + n1 + n2;
  //std::string n4_name("n4");
  //n4.SetName(n4_name);
  //std::cout << "n4: " << n4.DebugString() << std::endl; 
  
  Node xx = nx.XX(n3);
  LOG(INFO) << "xx: " << xx.DebugString();

  //print_inputs(n4);
  // node -operator

  return 0;
}
