; SPDX-License-Identifier: MIT
; Independent LLVM probes for runtime qualification during Loom GCN5 bringup.
; Build with clang -target amdgcn-amd-amdhsa -mcpu=gfx906 -nogpulib -O2 -c,
; then link the object with ld.lld -shared. No ROCr/HIP runtime is linked.
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @probe(ptr addrspace(1) %out) #0 {
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %idx = zext i32 %lane to i64
  %address = getelementptr i32, ptr addrspace(1) %out, i64 %idx
  store i32 42, ptr addrspace(1) %address, align 4
  ret void
}

define amdgpu_kernel void @transform(ptr addrspace(1) %input, ptr addrspace(1) %out, i32 %bias) #0 {
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %idx = zext i32 %lane to i64
  %source = getelementptr i32, ptr addrspace(1) %input, i64 %idx
  %destination = getelementptr i32, ptr addrspace(1) %out, i64 %idx
  %value = load i32, ptr addrspace(1) %source, align 4
  %scaled = mul i32 %value, 3
  %result = add i32 %scaled, %bias
  store i32 %result, ptr addrspace(1) %destination, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
attributes #0 = { "amdgpu-flat-work-group-size"="64,64" }
