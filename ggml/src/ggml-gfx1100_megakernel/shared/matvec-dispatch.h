// shared/matvec-dispatch.h — declarations only
#pragma once
bool is_float_type(int type);
hipFunction_t pick_matvec(int type);
hipFunction_t pick_matvec_res(int type);
void launch_matvec_typed(int type, const void * w, long long st,
                          float * output, int in_dim, int out_dim, hipStream_t stream);
void launch_matvec_res_typed(int type, const void * w, long long st,
                              float * residual, float * output, int in_dim, int out_dim,
                              hipStream_t stream);
void quant_and_launch_matvec(int type, const void * w, long long st,
                              float * input_f32, float * output,
                              int in_dim, int out_dim, hipStream_t stream);
