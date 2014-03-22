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

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

; the initial value of xxx is 10
@xxx = internal addrspace(1) global i32 10, align 4

; the initial value of yyy is 100
@yyy = internal addrspace(1) global i32 100, align 4

@llvm.used = appending global [3 x i8*] [i8* bitcast (i32 addrspace(1)* @xxx to i8*), i8* bitcast (i32 addrspace(1)* @yyy to i8*), i8* bitcast (void (i32*)* @test_kernel to i8*)], section "llvm.metadata"

; %ptr can be in the managed space, and its address can be directly used in the host and device.
; See the uvmlite.c, which passes the device pointer of xxx as the kernel parameter.
; This kernel also directly accesses @yyy, which is also managed. 
define void @test_kernel(i32* nocapture %ptr) nounwind alwaysinline {
  ; *%ptr = *%ptr + 20
  %gen2other = tail call i32 addrspace(1)* @llvm.nvvm.ptr.gen.to.global.p1i32.p0i32(i32* %ptr)
  %tmp1 = load i32 addrspace(1)* %gen2other, align 4
  %add = add nsw i32 %tmp1, 20
  store i32 %add, i32 addrspace(1)* %gen2other, align 4

  ; @yyy = @yyy + 30
  %tmp2 = load i32 addrspace(1)* @yyy, align 4
  %add3 = add nsw i32 %tmp2, 30
  store i32 %add3, i32 addrspace(1)* @yyy, align 4
  ret void
}

declare i32 addrspace(1)* @llvm.nvvm.ptr.gen.to.global.p1i32.p0i32(i32*) nounwind readnone

!nvvm.annotations = !{!7, !8, !9}

!7 = metadata !{i32 addrspace(1)* @xxx, metadata !"managed", i32 1}
!8 = metadata !{i32 addrspace(1)* @yyy, metadata !"managed", i32 1}
!9 = metadata !{void (i32*)* @test_kernel, metadata !"kernel", i32 1}
