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

; This NVVM IR program shows how to use shared memory.
; What it does is similar to the following CUDA C code.
;
;
; __shared__ int a;
; __shared__ int b[10];
; 
; __global__ void foo()
; {
;   a = 1;
;   b[5] = 2;
; }
;
  
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

@a = internal addrspace(3) global i32 0, align 4
@b = internal addrspace(3) global [10 x i32] zeroinitializer, align 4

define void @foo() {
entry:
  store i32 1, i32 addrspace(3)* @a, align 4
  store i32 2, i32 addrspace(3)* getelementptr inbounds ([10 x i32] addrspace(3)* @b, i64 0, i64 5), align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = metadata !{void ()* @foo, metadata !"kernel", i32 1}
