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

; This NVVM IR program shows how to call the malloc and free functions. 
; What it does is similar to the following CUDA C code.
;
; __device__ int *p;
; 
; __global__ void foo()
; {
;   p = (int *)malloc(128);
;   free(p);
; }
;

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

@p = internal addrspace(1) global i32* null, align 8

define void @foo() {
entry:
  %call = tail call i8* @malloc(i64 128)
  %conv = bitcast i8* %call to i32*
  store i32* %conv, i32* addrspace(1)* @p, align 8
  tail call void @free(i8* %call)
  ret void
}

declare noalias i8* @malloc(i64) nounwind
declare void @free(i8* nocapture) nounwind

!nvvm.annotations = !{!0}
!0 = metadata !{void ()* @foo, metadata !"kernel", i32 1}
