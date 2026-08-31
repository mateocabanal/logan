#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

/*
 * CPU-only dense BF16 GEMV through BNNS. The BF16 weight storage remains
 * caller-owned; BNNS receives a descriptor over that exact allocation, so
 * enabling this path cannot recreate Logan's old multi-GiB GDN weight copy.
 * Workspace is thread-local and grows monotonically, avoiding per-call
 * allocation/free in BNNSMatMul.
 */
extern "C" int coli_bnns_bf16_matmul(const uint16_t *w,
                                      const float *x,
                                      float *y,
                                      int O,
                                      int I) {
    if (!w || !x || !y || O <= 0 || I <= 0) return 0;

    BNNSNDArrayDescriptor a = {};
    a.flags = 0;
    a.layout = BNNSDataLayoutRowMajorMatrix;
    a.size[0] = (size_t)I;
    a.size[1] = 1;
    a.stride[0] = 1;
    a.stride[1] = (size_t)I;
    a.data = (void *)x;
    a.data_type = BNNSDataTypeFloat32;
    a.data_scale = 1.0f;
    a.data_bias = 0.0f;

    BNNSNDArrayDescriptor b = {};
    b.flags = 0;
    b.layout = BNNSDataLayoutRowMajorMatrix;
    b.size[0] = (size_t)I;
    b.size[1] = (size_t)O;
    b.stride[0] = 1;
    b.stride[1] = (size_t)I;
    b.data = (void *)w;
    b.data_type = BNNSDataTypeBFloat16;
    b.data_scale = 1.0f;
    b.data_bias = 0.0f;

    BNNSNDArrayDescriptor c = {};
    c.flags = 0;
    c.layout = BNNSDataLayoutRowMajorMatrix;
    c.size[0] = (size_t)O;
    c.size[1] = 1;
    c.stride[0] = 1;
    c.stride[1] = (size_t)O;
    c.data = (void *)y;
    c.data_type = BNNSDataTypeFloat32;
    c.data_scale = 1.0f;
    c.data_bias = 0.0f;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    const size_t need = BNNSMatMulWorkspaceSize(false, true, 1.0f, &a, &b, &c, nullptr);
    static thread_local std::vector<uint8_t> workspace;
    if (workspace.size() < need) workspace.resize(need);
    void *ws = need ? workspace.data() : nullptr;
    const int rc = BNNSMatMul(false, true, 1.0f, &a, &b, &c, ws, nullptr);
#pragma clang diagnostic pop
    return rc == 0 ? 1 : 0;
}
