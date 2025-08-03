#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
GGML_API GGML_DEPRECATED(ggml_backend_t ggml_backend_opencl_init(void), "use ggml_backend_dev_init() instead");

GGML_API bool ggml_backend_is_opencl(ggml_backend_t backend);

//
// device API
//
GGML_API ggml_backend_reg_t ggml_backend_opencl_reg(void);

#ifdef  __cplusplus
}
#endif
