#include <cute/config.hpp>
#include <cute/layout.hpp>
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
    return 0;
}