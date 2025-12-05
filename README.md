# Emotion Classification with LoRA Fine-Tuning

**Author:** El Brewster

**Base Model:** `distilroberta-base`  
**Dataset:** [Emotion Dataset (dair-ai/emotion)](https://huggingface.co/datasets/dair-ai/emotion)

---

## Project Summary

This project demonstrates **parameter-efficient fine-tuning (PEFT)** using **LoRA (Low-Rank Adaptation)** on a pre-trained transformer model for text classification.  
The model is fine-tuned to classify short texts (tweets) into six emotions: **anger, fear, joy, love, sadness, surprise**.

---

## LoRA Configuration

| Parameter       | Value |
|-----------------|-------|
| `lora_r`        | 8     |
| `lora_alpha`    | 16    |
| `lora_dropout`  | 0.1   |
| `task_type`     | SEQ_CLS (Sequence Classification) |
| `inference_mode`| False |

> Only the low-rank adaptation matrices (LoRA parameters) are trainable, keeping the vast majority of the original model frozen.

---

## Evaluation Metrics on Test Set

| Metric   | Score  |
|----------|--------|
| Accuracy | 0.914  |
| F1-Score | 0.874  |

---

## Why LoRA Saves Computational Resources

LoRA reduces the number of trainable parameters by **injecting small, low-rank matrices** into each layer of the pre-trained model instead of updating all model weights.  
- Only a fraction of the model’s parameters are updated.  
- Memory usage and GPU compute requirements are significantly lower.  
- Training is faster and allows fine-tuning large models on limited hardware.

This makes LoRA ideal for practical transfer learning when computational resources are constrained.

---

## References

- Hugging Face Transformers: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)  
- Hugging Face PEFT & LoRA: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)  
- Emotion Dataset: [https://huggingface.co/datasets/dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)

## Note (December 5)
During training, my dataset’s `label` values were stored as `torch.tensor` objects rather than plain Python integers, which caused `set(ds["train"]["label"])` to treat every tensor as a unique object instead of grouping them by value. As a result, the code incorrectly inferred that the dataset contained ~16,000 distinct labels, so the model was initialized with a 16k-dimensional classification head. This led to a mismatch when loading the LoRA adapter later, because the saved adapter expected 16,000 output classes while the correct model only has 6.
