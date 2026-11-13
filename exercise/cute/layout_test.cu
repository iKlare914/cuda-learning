#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/stride.hpp>
#include <cute/util/print_tensor.hpp>
#include <cute/util/print_latex.hpp>
#include <iostream>
using namespace cute;

int main() {
    Layout layout_a = make_layout(make_shape(Int<2>{}, make_shape(Int<2>{}, Int<2>{}, Int<2>{})), make_stride(Int<2>{}, make_stride(Int<4>{}, Int<1>{}, Int<1>{})));
    Layout layout_b = make_layout(make_shape(make_shape(Int<2>{}, Int<2>{}, Int<2>{}), Int<2>{}), make_stride(make_stride(Int<4>{}, Int<1>{}, Int<1>{}), Int<2>{}));
    std::cout << "===A===" << std::endl;
    print_layout(layout_a);
    std::cout << "===B===" << std::endl;
    print_layout(layout_b);
    std::cout << "idx2crd" << std::endl;
    auto shape = Shape<_3,Shape<_2,_3>>{};
    print(idx2crd(   16, shape));                                // (1,(1,2))
    print(idx2crd(_16{}, shape));                                // (_1,(_1,_2))
    print(idx2crd(make_coord(   1,5), shape));                   // (1,(1,2))
    print(idx2crd(make_coord(_1{},5), shape));                   // (_1,(1,2))
    print(idx2crd(make_coord(   1,make_coord(1,   2)), shape));  // (1,(1,2))
    print(idx2crd(make_coord(_1{},make_coord(1,_2{})), shape));  // (_1,(1,_2))
    return 0;
}