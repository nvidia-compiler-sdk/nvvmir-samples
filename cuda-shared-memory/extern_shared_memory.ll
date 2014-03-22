; Copyright (c) 2014 NVIDIA Corporation
;
; Permission is hereby granted, free of charge, to any person obtaining a copy
; of this software and associated documentation files (the "Software"), to deal
; in the Software without restriction, including without limitation the rights
; to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
; copies of the Software, and to permit persons to whom the Software is
; furnished to do so, subject to the following conditions:
;
; The above copyright notice and this permission notice shall be included in
; all copies or substantial portions of the Software.
;
; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
; IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
; FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
; AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
; LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
; OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
; SOFTWARE.

; This NVVM IR program shows how to use unsize external shared memory.
; What it does is similar to the following CUDA C code.
;
; extern __shared__ int a[];
;
; __global__ void foo()
; {
;    int *p1 = a;
;    float *p2 = (float *)&a[10];
;
;   *p1 = 10;
;   *p2 = 21.0f;
; }
;

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

@a = external addrspace(3) global [0 x i32]

define void @foo() {
entry:
  store i32 10, i32 addrspace(3)* getelementptr inbounds ([0 x i32] addrspace(3)* @a, i64 0, i64 0), align 4
  store float 2.100000e+01, float addrspace(3)* bitcast (i32 addrspace(3)* getelementptr inbounds ([0 x i32] addrspace(3)* @a, i64 0, i64 10) to float addrspace(3)*), align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = metadata !{void ()* @foo, metadata !"kernel", i32 1}
