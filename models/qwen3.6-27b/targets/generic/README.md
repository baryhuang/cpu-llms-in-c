# Qwen3.6-27B generic target

Status: graph and packing contract only. A portable full-model C runtime has not been emitted.

This directory is the target-independent fallback. It will own scalar reference kernels and the C graph executor after checkpoint import is pinned. It must not contain Apple Metal, AMX, Accelerate, or machine-topology assumptions.
