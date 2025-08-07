# Questions on Fine Tuning

## 1. What is Fine-tuning?

**Answer:**
Fine-tuning is the process of adapting a pre-trained model to a specific task or domain by continuing training on task-specific data with typically lower learning rates.

**Purpose:** Leverages general knowledge from pre-training while specializing for specific applications like sentiment analysis, domain-specific text generation, or task-specific classification.

**Benefits:** Faster training, better performance than training from scratch, requires less data, and maintains general capabilities while gaining specialized skills.

## 2. Describe the Fine-tuning process.

**Answer:**
**Steps:** Load pre-trained model → Prepare task-specific dataset → Adjust architecture (if needed) → Set lower learning rate → Train on task data → Evaluate performance.

**Key Considerations:** Learning rate scheduling, avoiding catastrophic forgetting, monitoring both task performance and general capabilities.

**Variants:** Full fine-tuning (update all parameters), partial fine-tuning (freeze some layers), parameter-efficient methods (LoRA, adapters).

## 3. What are the different Fine-tuning methods?

**Answer:**
**Full Fine-tuning:** Update all model parameters, highest performance but requires significant compute.

**Parameter-Efficient Fine-tuning (PEFT):** LoRA, adapters, prefix tuning - update minimal parameters while maintaining performance.

**Instruction Tuning:** Train on instruction-following datasets to improve task generalization.

**Task-specific Fine-tuning:** Adapt model for specific tasks like classification, summarization, or QA.

## 4. When should you go for fine-tuning?

**Answer:**
**Domain Mismatch:** When pre-trained model doesn't perform well on your specific domain or task.

**Task Specialization:** Need better performance on specific tasks than general-purpose models provide.

**Style Adaptation:** Require specific writing style, tone, or format consistency.

**Limited Data:** Have some task-specific data but not enough to train from scratch.

## 5. What is the difference between Fine-tuning and Transfer Learning?

**Answer:**
**Transfer Learning:** Broader concept of using knowledge from one task to improve performance on another task.

**Fine-tuning:** Specific technique within transfer learning that continues training pre-trained models on new data.

**Scope:** Transfer learning includes feature extraction, fine-tuning, and domain adaptation, while fine-tuning specifically refers to continued training.

**Usage:** Fine-tuning is one method to achieve transfer learning in neural networks.

## 6. Write about the instruction finetune and explain how does it work.

**Answer:**
Instruction fine-tuning trains models to follow natural language instructions by using datasets with instruction-response pairs.

**Process:** Collect diverse instruction datasets → Format as input-output pairs → Fine-tune with supervised learning → Evaluate instruction-following capability.

**Examples:** "Translate this to French: Hello" → "Bonjour", "Summarize this text: ..." → summary.

**Benefits:** Improves zero-shot task generalization and makes models more helpful for diverse user requests.

## 7. Explaining RLHF in Detail.

**Answer:**
RLHF (Reinforcement Learning from Human Feedback) aligns AI models with human preferences through a three-step process.

**Step 1:** Collect human preference data by having humans rank model outputs for quality and helpfulness.

**Step 2:** Train reward model to predict human preferences using the preference data.

**Step 3:** Use reinforcement learning (PPO) to optimize the language model policy to maximize the reward model scores.

**Result:** Models that are more helpful, harmless, and honest according to human judgment.

## 8. Write the different RLHF techniques.

**Answer:**
**PPO (Proximal Policy Optimization):** Most common method for policy optimization in RLHF.

**Constitutional AI:** Uses AI principles and self-critique rather than human feedback.

**DPO (Direct Preference Optimization):** Directly optimizes policy without separate reward model.

**RLAIF:** Uses AI feedback instead of human feedback for scalability.

**Red Team Training:** Adversarial training to improve safety and robustness.

## 9. Explaining PEFT in Detail.

**Answer:**
Parameter-Efficient Fine-Tuning (PEFT) adapts large models by updating only a small subset of parameters while keeping most weights frozen.

**Advantages:** Reduces computational cost, prevents catastrophic forgetting, enables multiple task adaptations, requires less storage.

**Methods:** LoRA (Low-Rank Adaptation), adapters, prefix tuning, prompt tuning, BitFit.

**Applications:** Multi-task learning, resource-constrained environments, rapid domain adaptation, personalization.

## 10. What is LoRA and QLoRA?

**Answer:**
**LoRA (Low-Rank Adaptation):** Decomposes weight updates into low-rank matrices, reducing trainable parameters by 99% while maintaining performance.

**QLoRA (Quantized LoRA):** Combines LoRA with 4-bit quantization, enabling fine-tuning of large models on consumer GPUs.

**Benefits:** Memory efficient, faster training, multiple adapters can be stored for different tasks, minimal performance degradation.

**Formula:** W = W₀ + BA where B and A are low-rank matrices with rank r << model dimension.

## 11. Define "pre-training" vs. "fine-tuning" in LLMs.

**Answer:**
**Pre-training:** Trains models on large, diverse text corpora using self-supervised objectives (next token prediction, masked language modeling).

**Fine-tuning:** Adapts pre-trained models to specific tasks using smaller, task-specific datasets with supervised learning.

**Scale:** Pre-training uses billions of tokens, fine-tuning uses thousands to millions of examples.

**Purpose:** Pre-training learns general language understanding, fine-tuning specializes for specific applications.

## 12. How do you train LLM models with billions of parameters?(training pipeline of llm)

**Answer:**
**Infrastructure:** Multi-GPU clusters with high-bandwidth interconnects (InfiniBand), distributed training frameworks.

**Parallelization:** Data parallelism, model parallelism, pipeline parallelism, tensor parallelism.

**Memory Optimization:** Gradient checkpointing, mixed precision training, ZeRO optimizer states.

**Training Techniques:** Gradient accumulation, learning rate scheduling, careful initialization, monitoring and checkpointing.

## 13. How does LoRA work?

**Answer:**
LoRA freezes original weights and adds trainable low-rank decomposition matrices to each layer.

**Mechanism:** For weight matrix W, add ΔW = B×A where A is r×d and B is d×r with r << d.

**Training:** Only A and B matrices are updated, dramatically reducing trainable parameters.

**Inference:** Merge LoRA weights with original weights or keep separate for multi-task scenarios.

## 14. How do you train an LLM model that prevents prompt hallucinations?

**Answer:**
**Data Quality:** Use high-quality, factually verified training data and remove low-quality sources.

**RLHF Training:** Train reward models to penalize hallucinated content and factual errors.

**Constitutional AI:** Teach models to self-critique and identify potential hallucinations.

**Retrieval Augmentation:** Combine with RAG systems to ground responses in factual sources.

## 15. How do you prevent bias and harmful prompt generation?

**Answer:**
**Data Curation:** Remove biased and harmful content from training data, use diverse and representative datasets.

**Safety Training:** Constitutional AI, RLHF with safety-focused reward models, red team evaluation.

**Content Filtering:** Input and output filtering systems, bias detection tools.

**Evaluation:** Regular bias auditing, fairness metrics, diverse evaluation teams.

## 16. How does proximal policy gradient work in a prompt generation?

**Answer:**
PPO optimizes language model policy by limiting policy updates to prevent large changes that could destabilize training.

**Mechanism:** Uses clipping mechanism to bound policy ratio, maintains exploration while ensuring stable updates.

**Application:** In RLHF, PPO updates model to generate prompts that receive higher reward scores while staying close to original policy.

**Benefits:** Stable training, prevents policy collapse, balances exploration and exploitation.

## 17. How does knowledge distillation benefit LLMs?

**Answer:**
Knowledge distillation trains smaller "student" models to mimic larger "teacher" models, creating efficient deployable models.

**Benefits:** Reduced inference cost, faster response times, smaller memory footprint, maintained performance.

**Process:** Teacher model generates soft targets, student learns from both hard labels and teacher predictions.

**Applications:** Model compression, edge deployment, cost reduction while preserving capability.

## 18. What's "few-shot" learning in LLMs?(RAG)

**Answer:**
Few-shot learning enables models to perform new tasks with only a few examples provided in the prompt context.

**Mechanism:** In-context learning where examples demonstrate the task pattern without parameter updates.

**Advantage:** No training required, immediate adaptation, versatile task performance.

**Limitation:** Context length constraints, performance may be lower than fine-tuned models.

## 19. Evaluating LLM performance metrics?

**Answer:**
**Generation Quality:** Perplexity, BLEU, ROUGE, BERTScore, human evaluation.

**Task Performance:** Accuracy, F1-score, exact match for specific tasks.

**Safety Metrics:** Toxicity scores, bias measurements, harmful content detection.

**Efficiency:** Inference speed, memory usage, throughput, cost per token.

## 20. How would you use RLHF to train an LLM model?(RLHF)

**Answer:**
**Data Collection:** Gather human preferences on model outputs across diverse prompts and tasks.

**Reward Modeling:** Train neural network to predict human preferences using comparison data.

**Policy Optimization:** Use PPO to fine-tune language model to maximize reward model scores.

**Evaluation:** Test on held-out preferences, measure alignment with human judgment, assess safety improvements.

## 21. What techniques can be employed to improve the factual accuracy of text generated by LLMs?(RAGA)

**Answer:**
**Retrieval Augmentation:** RAG systems ground generation in factual sources.

**Fact-checking Integration:** Real-time verification against knowledge bases.

**Training Improvements:** High-quality factual datasets, constitutional AI for accuracy.

**Post-processing:** Automated fact verification, uncertainty estimation, source citation.

## 22. How would you detect drift in LLM performance over time, especially in real-world production settings?(monitoring and evaluation metrics)

**Answer:**
**Performance Monitoring:** Track key metrics (accuracy, user satisfaction) over time, detect statistical deviations.

**Data Distribution:** Monitor input distribution changes, detect domain shift.

**User Feedback:** Continuous collection of user ratings and corrections.

**Automated Testing:** Regular evaluation on benchmark datasets, A/B testing for model versions.

## 23. Describe strategies for curating a high-quality dataset tailored for training a generative AI model.

**Answer:**
**Source Selection:** Choose reputable, diverse sources with appropriate licensing.

**Quality Filtering:** Remove low-quality, duplicate, or biased content through automated and manual review.

**Data Validation:** Fact-checking, expert review for domain-specific content.

**Diversity Metrics:** Ensure representation across demographics, topics, and styles.

## 24. What methods exist to identify and address biases within training data that might impact the generated output?(eval metrics)

**Answer:**
**Bias Detection:** Statistical analysis of demographic representation, sentiment analysis across groups.

**Fairness Metrics:** Equalized odds, demographic parity, individual fairness measures.

**Mitigation:** Data augmentation for underrepresented groups, bias-aware sampling, counterfactual data augmentation.

**Evaluation:** Regular bias audits, diverse evaluation teams, fairness-aware metrics.

## 25. How would you fine-tune LLM for domain-specific purposes like financial and medical applications?

**Answer:**
**Domain Data:** Collect high-quality domain-specific text (medical literature, financial reports).

**Specialized Tokenization:** Domain-specific vocabulary and terminology handling.

**Compliance:** Ensure regulatory compliance (HIPAA for medical, SEC for financial).

**Expert Validation:** Domain expert review of outputs, specialized evaluation metrics.

## 26. Explain the algorithm architecture for LLAMA and other LLMs alike.

**Answer:**
**Transformer Architecture:** Decoder-only transformer with RMSNorm, SwiGLU activation, rotary positional embeddings.

**Training:** Autoregressive language modeling on diverse text corpora with careful data curation.

**Optimizations:** Gradient checkpointing, mixed precision, efficient attention implementations.

**Scaling:** Available in multiple sizes (7B, 13B, 30B, 65B parameters) for different use cases and computational constraints.

---

*These answers cover fine-tuning concepts, PEFT methods, and LLM training strategies. Focus on understanding the trade-offs between different approaches and practical implementation considerations.*