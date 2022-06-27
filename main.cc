/*!
 * \brief
 * \file main.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/container/base.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

void inference(tvm::runtime::Module& gmod,
               const std::unordered_map<std::string, std::string>& input_paths, DLDevice dev) {
  tvm::runtime::PackedFunc get_input_info = gmod.GetFunction("get_input_info");
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_input = gmod.GetFunction("get_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc get_num_outputs = gmod.GetFunction("get_num_outputs");
  tvm::runtime::PackedFunc get_num_inputs = gmod.GetFunction("get_num_inputs");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  auto input_info = static_cast<tvm::Map<tvm::String, tvm::ObjectRef>>(get_input_info());
  auto shape_info = tvm::Downcast<tvm::Map<tvm::String, tvm::ShapeTuple>>(input_info["shape"]);
  auto dtype_info = tvm::Downcast<tvm::Map<tvm::String, tvm::String>>(input_info["dtype"]);
  std::unordered_map<std::string, DLDataType> dtype_map = {
      {"float32", DLDataType{kDLFloat, 32, 1}}, {"float16", DLDataType{kDLFloat, 16, 1}},
      {"int8", DLDataType{kDLInt, 8, 1}},       {"uint8", DLDataType{kDLUInt, 8, 1}},
      {"int16", DLDataType{kDLInt, 16, 1}},     {"uint16", DLDataType{kDLUInt, 16, 1}},
  };

  for (auto&& [name, path] : input_paths) {
    ICHECK(fs::exists(fs::path(path)) && fs::is_regular_file(path))
        << path << " is not exist or not a file";

    auto shape = shape_info[name];
    auto dtype = dtype_map[dtype_info[name]];

    size_t input_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) *
                        (dtype.bits / 8);
    std::ifstream input(path, std::ios::binary);
    auto x = tvm::runtime::NDArray::Empty(shape, dtype, dev);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
    ICHECK_EQ(buffer.size(), input_size);
    x.CopyFromBytes(buffer.data(), input_size);

    // set the input
    set_input(name, x);
  }

  LOG(INFO) << "Running graph executor...";

  // run the code
  run();
  // get the output
  tvm::runtime::NDArray output = get_output(0);

  if (const char* env_p = std::getenv("cpp_bench_debug");
      env_p != nullptr && strcmp(env_p, "ON") == 0) {
    size_t output_num =
        std::accumulate(output.Shape().begin(), output.Shape().end(), 1, std::multiplies<size_t>());
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < output_num / 10; i++) {
      printf("[%5d - %-5d]: ", i * 10, (i + 1) * 10 - 1);
      for (int j = 0; j < 10 && i * 10 + j < output_num; j++) {
        std::cout << ((float*)output->data)[i * 10 + j] << ", ";
      }
      std::cout << std::endl;
    }
  }
}

void evaluate(tvm::runtime::Module& gmod, DLDevice dev) {
  tvm::runtime::PackedFunc get_input_info = gmod.GetFunction("get_input_info");
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_input = gmod.GetFunction("get_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc get_num_outputs = gmod.GetFunction("get_num_outputs");
  tvm::runtime::PackedFunc get_num_inputs = gmod.GetFunction("get_num_inputs");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  tvm::runtime::PackedFunc time_eval =
      tvm::runtime::Registry::Get("runtime.RPCTimeEvaluator")
          ->
          operator()(gmod, "run", static_cast<int>(dev.device_type), dev.device_id, 10, 1, 500, "");

  int num_inputs = get_num_inputs();
  int num_outputs = get_num_outputs();
  int num_flat_args = num_inputs + num_outputs;
  std::unique_ptr<TVMValue> values(new TVMValue[num_flat_args]);
  std::unique_ptr<int> type_codes(new int[num_flat_args]);
  tvm::runtime::TVMArgsSetter setter(values.get(), type_codes.get());
  int offs = 0;

  for (int i = 0; i < num_inputs; i++) {
    DLTensor* arg =
        const_cast<DLTensor*>(static_cast<tvm::runtime::NDArray>(get_input(i)).operator->());
    setter(offs, arg);
    offs++;
  }
  for (uint32_t i = 0; i < num_outputs; ++i) {
    DLTensor* arg =
        const_cast<DLTensor*>(static_cast<tvm::runtime::NDArray>(get_output(i)).operator->());
    setter(offs, arg);
    offs++;
  }

  tvm::runtime::TVMRetValue rv;
  time_eval.CallPacked(tvm::runtime::TVMArgs(values.get(), type_codes.get(), num_flat_args), &rv);
  std::string results = rv.operator std::string();
  const double* results_arr = reinterpret_cast<const double*>(results.data());
  std::cout << results_arr[0] * 1000 << std::endl;
}

void DeployGraphExecutor(const std::string& module_path,
                         const std::unordered_map<std::string, std::string>& input_paths) {
  // load in the library
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(module_path);
  // create the graph executor module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);

  if (input_paths.empty())
    evaluate(gmod, dev);
  else
    inference(gmod, input_paths, dev);
}

int main(int argc, char* argv[]) {
  ICHECK_GE(argc, 2) << "usage: " << argv[0]
                     << " libmodel.so [input_name0:input0 input_name1:input1] ...";

  std::string module_path = argv[1];
  std::unordered_map<std::string, std::string> input_paths;

  for (int i = 2; i < argc; i++) {
    std::string tmp = argv[i];
    auto center = tmp.find(":");
    ICHECK_NE(center, std::string::npos) << "input format must be input_name:input";
    input_paths[tmp.substr(0, center)] = tmp.substr(center + 1);
  }

  DeployGraphExecutor(module_path, input_paths);
  return 0;
}
