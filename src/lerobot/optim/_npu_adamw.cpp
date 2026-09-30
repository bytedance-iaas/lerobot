#include <torch/extension.h>

#include <c10/core/DeviceGuard.h>
#include <op_plugin/utils/op_api_common_base.h>

void adamw(
    const std::vector<at::Tensor>& params,
    const std::vector<at::Tensor>& grads,
    const std::vector<at::Tensor>& exp_avgs,
    const std::vector<at::Tensor>& exp_avg_sqs,
    const std::vector<at::Tensor>& scalars) {
  TORCH_CHECK(
      params.size() == grads.size() && params.size() == exp_avgs.size() &&
          params.size() == exp_avg_sqs.size(),
      "AdamW tensor lists must have the same length");
  TORCH_CHECK(scalars.size() == 7, "AdamW requires seven scalar tensors");
  for (const auto& scalar : scalars) {
    TORCH_CHECK(scalar.dim() == 0, "AdamW scalar inputs must be zero-dimensional");
  }
  if (params.empty()) {
    return;
  }

  c10::DeviceGuard guard(params.front().device());
  const c10::optional<at::Tensor> max_grad_norm = c10::nullopt;
  const bool amsgrad = false;
  const bool maximize = false;
  for (size_t i = 0; i < params.size(); ++i) {
    EXEC_NPU_CMD_EXT(
        aclnnApplyAdamW,
        params[i],
        exp_avgs[i],
        exp_avg_sqs[i],
        scalars[0],
        scalars[1],
        scalars[2],
        scalars[3],
        scalars[4],
        scalars[5],
        scalars[6],
        grads[i],
        max_grad_norm,
        amsgrad,
        maximize);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("adamw", &adamw, pybind11::call_guard<pybind11::gil_scoped_release>());
}
