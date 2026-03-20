#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>

struct ResultRow {
    std::string backend;
    std::string mode;
    int batch_size;
    int in_features;
    int hidden_features;
    int out_features;
    float avg_ms;
    float tflops;
    int tf32;
};

static void checkCuda(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

static void checkCublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error %s: %d\n", msg, int(status));
        std::exit(EXIT_FAILURE);
    }
}

// d_h / d_y are treated as column-major matrices to match cuBLAS GEMM outputs:
// GEMM1 output d_h: [hidden_features, batch_size] (lda/ldc = n1)
// GEMM2 output d_y: [out_features, batch_size] (lda/ldc = n2)
__global__ void add_bias_relu_kernel(float* d_mat, const float* d_bias, int n_rows, int n_cols) {
    const int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_rows * n_cols;
    const int stride = gridDim.x * blockDim.x;

    for (int idx = idx0; idx < total; idx += stride) {
        // Column-major linear index: idx = row + col*n_rows
        const int row = idx % n_rows;
        const float v = d_mat[idx] + d_bias[row];
        d_mat[idx] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void add_bias_kernel(float* d_mat, const float* d_bias, int n_rows, int n_cols) {
    const int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_rows * n_cols;
    const int stride = gridDim.x * blockDim.x;

    for (int idx = idx0; idx < total; idx += stride) {
        // Column-major linear index: idx = row + col*n_rows
        const int row = idx % n_rows;
        d_mat[idx] += d_bias[row];
    }
}

float benchmark_gemm_pair(
    cublasHandle_t handle,
    int batch_size,
    int in_features,
    int hidden_features,
    int out_features,
    bool use_tf32,
    int warmup_iters,
    int timed_iters
) {
    // Dimensions:
    // X: [batch, in_features]
    // W1: [hidden_features, in_features]   (so GEMM is: (batch x in) * (in x hidden) -> (batch x hidden)
    // W2: [out_features, hidden_features] -> (batch x out)

    int m1 = batch_size;
    int k1 = in_features;
    int n1 = hidden_features;

    int m2 = batch_size;
    int k2 = hidden_features;
    int n2 = out_features;

    size_t bytes_x  = size_t(m1) * k1 * sizeof(float);
    size_t bytes_w1 = size_t(k1) * n1 * sizeof(float);
    size_t bytes_w2 = size_t(k2) * n2 * sizeof(float);
    size_t bytes_h  = size_t(m1) * n1 * sizeof(float);
    size_t bytes_y  = size_t(m2) * n2 * sizeof(float);
    size_t bytes_b1 = size_t(n1) * sizeof(float);
    size_t bytes_b2 = size_t(n2) * sizeof(float);

    float* d_x = nullptr;
    float* d_w1 = nullptr;
    float* d_w2 = nullptr;
    float* d_h = nullptr;
    float* d_y = nullptr;
    float* d_b1 = nullptr;
    float* d_b2 = nullptr;

    checkCuda(cudaMalloc(&d_x, bytes_x), "malloc d_x");
    checkCuda(cudaMalloc(&d_w1, bytes_w1), "malloc d_w1");
    checkCuda(cudaMalloc(&d_w2, bytes_w2), "malloc d_w2");
    checkCuda(cudaMalloc(&d_h, bytes_h), "malloc d_h");
    checkCuda(cudaMalloc(&d_y, bytes_y), "malloc d_y");
    checkCuda(cudaMalloc(&d_b1, bytes_b1), "malloc d_b1");
    checkCuda(cudaMalloc(&d_b2, bytes_b2), "malloc d_b2");

    // Initialize with some values (we don't care about exact data)
    const size_t max_bytes = std::max(
        bytes_x,
        std::max(bytes_w1, std::max(bytes_w2, std::max(bytes_b1, bytes_b2)))
    );
    std::vector<float> h_tmp(max_bytes / sizeof(float), 1.0f);
    checkCuda(cudaMemcpy(d_x, h_tmp.data(), bytes_x, cudaMemcpyHostToDevice), "memcpy x");
    checkCuda(cudaMemcpy(d_w1, h_tmp.data(), bytes_w1, cudaMemcpyHostToDevice), "memcpy w1");
    checkCuda(cudaMemcpy(d_w2, h_tmp.data(), bytes_w2, cudaMemcpyHostToDevice), "memcpy w2");
    checkCuda(cudaMemcpy(d_b1, h_tmp.data(), bytes_b1, cudaMemcpyHostToDevice), "memcpy b1");
    checkCuda(cudaMemcpy(d_b2, h_tmp.data(), bytes_b2, cudaMemcpyHostToDevice), "memcpy b2");

    float alpha = 1.0f;
    float beta = 0.0f;

    // Baseline: cublasSgemm (FP32).
    // TF32 path: cublasGemmEx (FP32 input/output, TF32 compute on Ampere+).
    checkCublas(
        cublasSetMathMode(handle, use_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH),
        "set math mode"
    );

    const cublasOperation_t opN = CUBLAS_OP_N;
    const cudaDataType_t dt = CUDA_R_32F;

    // Launch configuration for bias+ReLU (GEMM1 output) and bias add (GEMM2 output).
    const int relu_threads = 256;
    const int relu_total = n1 * m1;
    const int relu_blocks = (relu_total + relu_threads - 1) / relu_threads;
    const int bias_total = n2 * m2;
    const int bias_blocks = (bias_total + relu_threads - 1) / relu_threads;

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        // Full FC forward:
        //   H = GEMM(X, W1) + b1
        //   H = ReLU(H)
        //   Y = GEMM(H, W2) + b2
        if (!use_tf32) {
            checkCublas(
                cublasSgemm(
                    handle,
                    opN, opN,
                    n1,              // m
                    m1,              // n
                    k1,              // k
                    &alpha,
                    d_w1, n1,        // A: [n1, k1]
                    d_x, k1,         // B: [k1, m1]
                    &beta,
                    d_h, n1          // C: [n1, m1]
                ),
                "warmup gemm1 sgemm"
            );
        } else {
            checkCublas(
                cublasGemmEx(
                    handle,
                    opN, opN,
                    n1, m1, k1,
                    &alpha,
                    d_w1, dt, n1,
                    d_x, dt, k1,
                    &beta,
                    d_h, dt, n1,
                    CUBLAS_COMPUTE_32F_FAST_TF32,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ),
                "warmup gemm1 gemmex tf32"
            );
        }

        add_bias_relu_kernel<<<relu_blocks, relu_threads>>>(d_h, d_b1, n1, m1);
        checkCuda(cudaGetLastError(), "warmup bias+relu kernel");

        if (!use_tf32) {
            checkCublas(
                cublasSgemm(
                    handle,
                    opN, opN,
                    n2,
                    m2,
                    k2,
                    &alpha,
                    d_w2, n2,
                    d_h, k2,
                    &beta,
                    d_y, n2
                ),
                "warmup gemm2 sgemm"
            );
        } else {
            checkCublas(
                cublasGemmEx(
                    handle,
                    opN, opN,
                    n2, m2, k2,
                    &alpha,
                    d_w2, dt, n2,
                    d_h, dt, k2,
                    &beta,
                    d_y, dt, n2,
                    CUBLAS_COMPUTE_32F_FAST_TF32,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ),
                "warmup gemm2 gemmex tf32"
            );
        }

        add_bias_kernel<<<bias_blocks, relu_threads>>>(d_y, d_b2, n2, m2);
        checkCuda(cudaGetLastError(), "warmup bias add kernel");
    }

    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    checkCuda(cudaEventRecord(start), "record start");

    for (int i = 0; i < timed_iters; ++i) {
        if (!use_tf32) {
            checkCublas(
                cublasSgemm(
                    handle,
                    opN, opN,
                    n1,
                    m1,
                    k1,
                    &alpha,
                    d_w1, n1,
                    d_x, k1,
                    &beta,
                    d_h, n1
                ),
                "timed gemm1 sgemm"
            );
        } else {
            checkCublas(
                cublasGemmEx(
                    handle,
                    opN, opN,
                    n1, m1, k1,
                    &alpha,
                    d_w1, dt, n1,
                    d_x, dt, k1,
                    &beta,
                    d_h, dt, n1,
                    CUBLAS_COMPUTE_32F_FAST_TF32,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ),
                "timed gemm1 gemmex tf32"
            );
        }

        add_bias_relu_kernel<<<relu_blocks, relu_threads>>>(d_h, d_b1, n1, m1);
        checkCuda(cudaGetLastError(), "timed bias+relu kernel");

        if (!use_tf32) {
            checkCublas(
                cublasSgemm(
                    handle,
                    opN, opN,
                    n2,
                    m2,
                    k2,
                    &alpha,
                    d_w2, n2,
                    d_h, k2,
                    &beta,
                    d_y, n2
                ),
                "timed gemm2 sgemm"
            );
        } else {
            checkCublas(
                cublasGemmEx(
                    handle,
                    opN, opN,
                    n2, m2, k2,
                    &alpha,
                    d_w2, dt, n2,
                    d_h, dt, k2,
                    &beta,
                    d_y, dt, n2,
                    CUBLAS_COMPUTE_32F_FAST_TF32,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ),
                "timed gemm2 gemmex tf32"
            );
        }

        add_bias_kernel<<<bias_blocks, relu_threads>>>(d_y, d_b2, n2, m2);
        checkCuda(cudaGetLastError(), "timed bias add kernel");
    }

    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "sync stop");

    float total_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&total_ms, start, stop), "elapsed time");

    float avg_ms = total_ms / timed_iters;

    // FLOPs: two GEMMs
    double flops = 2.0 * double(batch_size) * double(in_features) * double(hidden_features)
                 + 2.0 * double(batch_size) * double(hidden_features) * double(out_features);
    double t_sec = avg_ms / 1e3;
    double tflops = (flops / t_sec) / 1e12;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_h);
    cudaFree(d_y);
    cudaFree(d_b1);
    cudaFree(d_b2);

    return static_cast<float>(tflops);
}

int main(int argc, char** argv) {
    // Simple command-line parsing
    int batch_size = 256;
    int in_features = 4096;
    int out_features = 4096;
    std::vector<int> sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int warmup_iters = 25;
    int timed_iters = 200;
    std::string output_path = "results/cublas_benchmark.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::atoi(argv[++i]);
        } else if (arg == "--in_features" && i + 1 < argc) {
            in_features = std::atoi(argv[++i]);
        } else if (arg == "--out_features" && i + 1 < argc) {
            out_features = std::atoi(argv[++i]);
        } else if (arg == "--sizes" && i + 1 < argc) {
            sizes.clear();
            std::string s = argv[++i];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                if (comma == std::string::npos) comma = s.size();
                int v = std::atoi(s.substr(pos, comma - pos).c_str());
                if (v > 0) sizes.push_back(v);
                pos = comma + 1;
            }
        } else if (arg == "--warmup_iters" && i + 1 < argc) {
            warmup_iters = std::atoi(argv[++i]);
        } else if (arg == "--timed_iters" && i + 1 < argc) {
            timed_iters = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "create handle");

    std::vector<ResultRow> results;

    for (int use_tf32_int = 0; use_tf32_int < 2; ++use_tf32_int) {
        bool use_tf32 = (use_tf32_int == 1);
        std::string mode = use_tf32 ? "tensor_core_tf32" : "cuda_core_fp32";

        // Baseline uses cublasSgemm (FP32).
        // TF32 path uses cublasGemmEx (FP32 input/output, TF32 compute on Ampere+).

        for (int hidden : sizes) {
            // Time the GEMM pair
            float tflops = benchmark_gemm_pair(
                handle,
                batch_size,
                in_features,
                hidden,
                out_features,
                use_tf32,
                warmup_iters,
                timed_iters
            );

            // Recompute avg_ms from FLOPs and TFLOPs
            double flops = 2.0 * double(batch_size) * double(in_features) * double(hidden)
                         + 2.0 * double(batch_size) * double(hidden) * double(out_features);
            double tflops_d = tflops;
            double t_sec = (flops / 1e12) / tflops_d;
            float avg_ms = static_cast<float>(t_sec * 1e3);

            results.push_back(
                {
                    "cublas",
                    mode,
                    batch_size,
                    in_features,
                    hidden,
                    out_features,
                    avg_ms,
                    tflops,
                    use_tf32 ? 1 : 0
                }
            );

            std::printf(
                "mode=%s hidden=%d avg_ms=%.3f TFLOPs=%.2f\n",
                mode.c_str(), hidden, avg_ms, tflops
            );
        }
    }

    cublasDestroy(handle);

    // Write CSV header compatible with plotting script
    try {
        std::filesystem::path outp(output_path);
        if (outp.has_parent_path()) {
            std::filesystem::create_directories(outp.parent_path());
        }
    } catch (...) {
        // If directory creation fails, the file open below will fail and report an error.
    }

    std::ofstream ofs(output_path);
    if (!ofs) {
        std::fprintf(stderr, "Failed to open output file %s\n", output_path.c_str());
        return EXIT_FAILURE;
    }

    ofs << "backend,mode,batch_size,in_features,hidden_features,out_features,avg_ms,tflops,tf32\n";
    for (const auto& r : results) {
        ofs << r.backend << ","
            << r.mode << ","
            << r.batch_size << ","
            << r.in_features << ","
            << r.hidden_features << ","
            << r.out_features << ","
            << r.avg_ms << ","
            << r.tflops << ","
            << r.tf32
            << "\n";
    }
    ofs.close();

    std::printf("Wrote %zu rows to %s\n", results.size(), output_path.c_str());
    return 0;
}

