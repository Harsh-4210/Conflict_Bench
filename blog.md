# Post-Mortem: Teaching LLMs to Resolve Business Conflicts with GRPO

In high-stakes business environments, instructions are often contradictory. A CEO orders a hiring freeze while a Project Manager demands 5 new hires to meet a deadline. Currently, resolving these conflicts requires expensive human oversight. 

**ConflictBench** is an experimental framework designed to automate this resolution using **Group Relative Policy Optimization (GRPO)**.

## The Challenge: Authority-Aware Reasoning
Standard LLMs often struggle with "who to listen to" when faced with conflicting prompts. They tend to be overly agreeable (the "yes-man" problem) or simply pick the most recent instruction. 

We trained **ConflictBench-v1** (based on Qwen-2.5-3B) to understand **Authority Hierarchies**.

## The Results: Run 2 Highlights
In our second major training run ("The Hardening"), we observed several critical behaviors:

### 1. The Reward Breakthrough
We saw a monotonic improvement in reward from **0.37 to 0.49** in just 500 steps. This proves the model successfully learned the deterministic rules of our authority verifier.

### 2. Emergent Efficiency (Inverse Scaling)
Intriguingly, we discovered a negative correlation between reasoning length and reward. The model learned that **clear, punchy logic** was more effective at resolving conflicts than long, verbose "thought" blocks. 

### 3. Safety First
By monitoring KL Divergence, we ensured the model remained an excellent instruction-follower while gaining specialized conflict-resolution skills.

## Conclusion
ConflictBench demonstrates that O1-style "thinking" models don't just need to be good at math or code—they can be trained to handle the messy, contradictory realities of corporate operations.

[Explore the Code](https://github.com/Harsh-4210/Conflict_Bench) | [Try the Demo](https://huggingface.co/spaces/Harsh-4210/ConflictBench)
