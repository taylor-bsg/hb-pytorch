#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/utils/ParamUtils.h>

namespace at { 
namespace native {

Tensor dense_to_cudalite(const Tensor& cpu_tensor) {
  AT_ASSERTM(cpu_tensor.device().type() == DeviceType::CPU,
             "dense_to_cudalite expects CPU tensor input");
  AT_ASSERTM(cpu_tensor.layout() == Layout::Strided,
             "dense_to_cudalite expects strided tensor input");
  auto cpu_tensor_cont = cpu_tensor.contiguous();
  Tensor cudalite_tensor = at::native::empty_cudalite(cpu_tensor_cont.sizes(), cpu_tensor_cont.options());
  // TODO: this is incomplete -> data is not copied over!
  return cudalite_tensor;
}

}} // at::native
