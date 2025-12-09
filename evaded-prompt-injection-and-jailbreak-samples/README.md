---
license: cc-by-nc-4.0
task_categories:
- text-classification
- text-generation
language:
- en
pretty_name: Evaded Prompt Injection and Jailbreak Samples
size_categories:
- 1K<n<10K
---

This dataset originates from our paper '[Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails](https://arxiv.org/abs/2504.11168)'.

The dataset contains a mixture of prompt injections and jailbreak samples modified via character injection and adversarial ML evasion techniques (Techniques can be found within the paper above). For each sample we provide the original unaltered prompt and a modified prompt, the `attack_name` outlines which attack technique was used to modify the sample.

### Acknowledgements

The original prompt injection samples were curated from [Safe-Guard-Prompt-Injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection).

### Additional Notes

- Emoji Smuggling modified prompts are stored as Base64 to ensure the original encoding is retained.
- There are mixture of injected unicode characters which may cause some issues when dumped from JSON.

### Links

[Mindgard](https://mindgard.ai/)

[ArXiv Paper](https://arxiv.org/abs/2504.11168)