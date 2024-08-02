
#include <cxxopts.hpp>
#include <nlohmann/json.hpp>
#include <NvInferRuntime.h>
#include <iostream>
#include <fstream>
#include <filesystem>

using json = nlohmann::json;
using fs = std::filesystem;

enum Error{
    Success = 0,
    FileNotExist = 1,
    ParseError = 2,
    ConfigNotFound = 3,
    EngineNotFound = 4;
};

struct Sample
{
    fs::path engine_dir;
}

int main(int argc, char* argv[])
{
    std::string engine_dir, input_data;
    try {
        cxxopts::Options options(
        "TensorRT Drive LLM Runtime", "TensorRT Auto Deployment Sample C++ Runtime.");
        options.add_options()("h,help", "Print usage");
        options.add_options()("engine_dir", "Directory that store the TRT engine and config.", cxxopts::value<std::string>());
        options.add_options()("input_data", "Json that stores the tokenized input tensors.",cxxopts::value<std::string>());
        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        // Argument: Engine directory
        if (!result.count("engine_dir"))
        {
            std::cout << options.help() << std::endl;
            std::cout << "Please specify engine directory." << std::endl;
            return FileNotExist;
        }

        engine_dir = result["engine_dir"].as<std::string>();

        if (!result.count("input_data"))
        {
            std::cout << options.help() << std::endl;
            std::cout << "Please specify input_data directory." << std::endl;
            return FileNotExist;
        }
        input_data = result["input_data"].as<std::string>();
    } catch (const cxxopts::OptionException& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        return ParseError;
    }
    return Success;
}
