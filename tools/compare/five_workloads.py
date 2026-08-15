"""The five workload user texts shared by every stack measurement."""

WORKLOADS = [
    ("code short", "Write a C function int max2(int a, int b) that returns "
     "the larger value. Output only the function."),
    ("prose", "Explain in about 300 words how a hash table works, covering "
     "buckets, hashing and collisions."),
    ("code long", "Write a Python class LRUCache with get and put methods "
     "using a doubly linked list and a dict, with O(1) operations. Include "
     "docstrings."),
    ("essay", "Write a detailed technical essay of about 1200 words on how "
     "operating systems manage virtual memory: paging, page tables, TLBs, "
     "page faults, swapping, and copy-on-write."),
    ("summary", "Summarize the following notes into a short paragraph. The "
     "first note says the deployment pipeline compiles model weights into "
     "memory-mapped images with SHA-256 pinning at every stage. The second "
     "note says the runtime is written in C and Metal with no Python in the "
     "deployed path. The third note says batched prefill processes prompt "
     "tokens through compiled graphs in chunks of sixty-four, thirty-two "
     "and sixteen. The fourth note says speculative decoding drafts tokens "
     "with a small head and verifies them in batched forwards, keeping "
     "output identical to plain greedy decoding. The fifth note says the "
     "attention key-value cache is stored in half precision to halve "
     "memory. The sixth note says performance is pinned to one machine and "
     "results do not transfer."),
]

WARMUP = "What is 2+2? Answer with only the number."


def render(user_text):
    return ("<|im_start|>user\n" + user_text + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
