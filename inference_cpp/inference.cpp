/**
 * C++ TorchScript 模型推理示例
 * 基于 LibTorch (PyTorch C++ API) 实现静态模型的加载和推理
 */

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

class ModelInference {
public:
    ModelInference(const std::string& model_path) {
        try {
            // 加载 TorchScript 模型
            std::cout << "正在加载模型: " << model_path << std::endl;
            module = torch::jit::load(model_path);
            module.eval();
            std::cout << "✓ 模型加载成功!" << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "错误: 无法加载模型\n" << e.what() << std::endl;
            throw;
        }
    }

    // 推理函数（通用版本）
    torch::Tensor infer(const torch::Tensor& input) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        
        // 禁用梯度计算以加速推理
        torch::NoGradGuard no_grad;
        
        // 执行推理
        auto output = module.forward(inputs).toTensor();
        return output;
    }

    // 推理函数（带时间统计）
    std::pair<torch::Tensor, double> infer_with_timing(const torch::Tensor& input, int warmup=5, int runs=100) {
        // 预热
        for (int i = 0; i < warmup; i++) {
            infer(input);
        }

        // 正式计时
        auto start = std::chrono::high_resolution_clock::now();
        torch::Tensor output;
        for (int i = 0; i < runs; i++) {
            output = infer(input);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time_ms = duration.count() / (runs * 1000.0);
        
        return {output, avg_time_ms};
    }

    // 获取模型信息
    void print_model_info() {
        std::cout << "\n=== 模型信息 ===" << std::endl;
        // 注意：TorchScript 模型的详细信息需要通过 graph 获取
        std::cout << "模型类型: TorchScript (ScriptModule)" << std::endl;
    }

private:
    torch::jit::script::Module module;
};

// 辅助函数：打印张量信息
void print_tensor_info(const torch::Tensor& tensor, const std::string& name) {
    std::cout << "\n" << name << ":" << std::endl;
    std::cout << "  形状: [";
    for (int i = 0; i < tensor.dim(); i++) {
        std::cout << tensor.size(i);
        if (i < tensor.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  数据类型: " << tensor.dtype() << std::endl;
    
    // 打印前几个元素
    auto flat = tensor.flatten();
    int num_elements = std::min(5, (int)flat.size(0));
    std::cout << "  前 " << num_elements << " 个元素: [";
    for (int i = 0; i < num_elements; i++) {
        std::cout << std::fixed << std::setprecision(6) << flat[i].item<float>();
        if (i < num_elements - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <模型路径> [输入形状]" << std::endl;
        std::cerr << "示例: " << argv[0] << " ../traced_models/simple_cnn_traced.pt 1,3,32,32" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    
    // 解析输入形状（默认: 1,3,32,32）
    std::vector<int64_t> input_shape = {1, 3, 32, 32};
    if (argc >= 3) {
        std::string shape_str = argv[2];
        input_shape.clear();
        size_t pos = 0;
        while ((pos = shape_str.find(',')) != std::string::npos) {
            input_shape.push_back(std::stoi(shape_str.substr(0, pos)));
            shape_str.erase(0, pos + 1);
        }
        input_shape.push_back(std::stoi(shape_str));
    }

    try {
        // 创建推理器
        ModelInference inference(model_path);
        inference.print_model_info();

        // 创建输入张量
        std::cout << "\n=== 创建输入张量 ===" << std::endl;
        std::cout << "输入形状: [";
        for (size_t i = 0; i < input_shape.size(); i++) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        auto input = torch::randn(input_shape);
        print_tensor_info(input, "输入张量");

        // 执行推理
        std::cout << "\n=== 执行推理 ===" << std::endl;
        auto [output, avg_time] = inference.infer_with_timing(input, 5, 100);
        
        print_tensor_info(output, "输出张量");
        
        std::cout << "\n=== 性能统计 ===" << std::endl;
        std::cout << "平均推理时间: " << std::fixed << std::setprecision(3) 
                  << avg_time << " ms" << std::endl;
        std::cout << "吞吐量: " << std::fixed << std::setprecision(2) 
                  << (1000.0 / avg_time) << " FPS" << std::endl;

        std::cout << "\n✓ C++ 推理完成!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
