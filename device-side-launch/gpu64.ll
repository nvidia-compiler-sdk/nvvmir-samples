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

; This NVVM IR program shows how to call cudaGetParameterBuffer and cudaLaunchDevice functions.
; What it does is similar to the following CUDA C code.
;
; __global__ void kernel(int depth)
; {
;   if (threadIdx.x == 0) {
;     printf("kernel launched, depth = %d\n", depth);
;   }
;  
;   __syncthreads();
;
;   if (++depth > 3) 
;     return;  
;      
;   kernel<<<1,1>>>(depth);
;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

@"$str" = private addrspace(1) constant [29 x i8] c"kernel launched, depth = %d\0A\00"

define void @kernel(i32 %depth) {
entry:
  %tmp31 = alloca i32, align 8
  %gen2local = call i32 addrspace(5)* @llvm.nvvm.ptr.gen.to.local.p5i32.p0i32(i32* %tmp31)
  %tmp31.sub = bitcast i32* %tmp31 to i8*
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = call i8* @llvm.nvvm.ptr.global.to.gen.p0i8.p1i8(i8 addrspace(1)* getelementptr inbounds ([29 x i8] addrspace(1)* @"$str", i64 0, i64 0))
  store i32 %depth, i32 addrspace(5)* %gen2local, align 8
  %call = call i32 @vprintf(i8* %1, i8* %tmp31.sub)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.cuda.syncthreads()
  %inc = add nsw i32 %depth, 1
  %cmp5 = icmp sgt i32 %inc, 3
  br i1 %cmp5, label %return, label %if.end7

if.end7:                                          ; preds = %if.end
  %call15 = call i8* @cudaGetParameterBufferV2(i8* bitcast (void (i32)* @kernel to i8*), %struct.dim3 { i32 1, i32 1, i32 1 }, %struct.dim3 { i32 1, i32 1, i32 1 }, i32 0)
  %tobool = icmp eq i8* %call15, null
  br i1 %tobool, label %return, label %cond.true

cond.true:                                        ; preds = %if.end7
  %conv = bitcast i8* %call15 to i32*
  store i32 %inc, i32* %conv, align 4
  %call20 = call i32 @cudaLaunchDeviceV2(i8* %call15, %struct.CUstream_st* null)
  br label %return

return:                                           ; preds = %cond.true, %if.end7, %if.end
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

declare i8* @llvm.nvvm.ptr.global.to.gen.p0i8.p1i8(i8 addrspace(1)*) nounwind readnone

declare i32 @vprintf(i8* nocapture, i8*) nounwind

declare void @llvm.cuda.syncthreads() nounwind

declare i8* @cudaGetParameterBufferV2(i8*, %struct.dim3, %struct.dim3, i32)

declare i32 @cudaLaunchDeviceV2(i8*, %struct.CUstream_st*)

declare i32 addrspace(5)* @llvm.nvvm.ptr.gen.to.local.p5i32.p0i32(i32*) nounwind readnone

!nvvm.annotations = !{!0}
!0 = metadata !{void (i32)* @kernel, metadata !"kernel", i32 1}
