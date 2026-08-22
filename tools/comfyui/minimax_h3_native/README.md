# MiniMax-H3 Native FL2VA for ComfyUI

This node uses ComfyUI only for image input, parameters, queueing, and output
display. The model itself runs in `build/minimax-h3-m3-e2e`; no PyTorch, MLX,
or Python tensor operation is used for inference.

For text-only generation, add **MiniMax-H3 Native Text to Video+Audio
(C/Metal)** under `cpullama/native` and enter the prompt directly in the node.
No `Load Image` node is required.

For image-conditioned generation, add `Load Image`, connect its `IMAGE` output
to **MiniMax-H3 Native FL2VA (C/Metal)**, then enter the prompt.

Use 128x128 and 22 frames for a fast pipeline test. The validated 480p geometry
is 864x480 and 124 frames; it is a long run on an M3 Pro.

The node returns the absolute MP4 path and the native metrics JSON. Outputs are
written under ComfyUI's configured output directory.
