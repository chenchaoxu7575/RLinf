Advanced Features
==============================

This chapter provides a step-by-step deep dive into how RLinf achieves **highly efficient execution**,  
offering practical guidance to help you fully optimize your RL post-training workflows.

- :doc:`5D`  
   Explains how RLinf supports Megatron-style 5D parallelism, including:  
   Tensor Parallelism (TP), Data Parallelism (DP), Pipeline Parallelism (PP),  
   Sequence Parallelism (SP), and Context Parallelism (CP).  
   Learn how to configure and combine these dimensions to scale large models efficiently.

- :doc:`lora`  
   Demonstrates how to integrate Low-Rank Adaptation (LoRA) into RLinf,  
   enabling parameter-efficient fine-tuning for large-scale models with minimal compute overhead.

- :doc:`version`  
   Describes how to dynamically switch between different SGLang versions  
   to accommodate varying compatibility needs or experimental requirements.

- :doc:`resume`  
   Covers how to resume training from saved checkpoints,  
   ensuring fault tolerance and seamless continuation for long-running or interrupted training jobs.

- :doc:`convertor`  
   Describes how to convert a saved checkpoint file into HuggingFace safetensors format,  
   which can be used for checkpoint evaluation or uploading to the HuggingFace Hub.

- :doc:`logger`  
   Introduces how to visualize and track key metrics during your training process.  
   Currently, we support three backends for experiment tracking and visualization:
   TensorBoard, Weights & Biases (wandb), and SwanLab.

- :doc:`weight_syncer`
   Introduces the actor-to-rollout weight synchronization optimization used in
   embodied training, including the ``patch`` and ``bucket`` modes, their
   configuration, recommended use cases, and performance considerations.

- :doc:`nsight`
   Introduces the Hydra-based ``cluster.nsight`` configuration used to wrap
   selected Ray worker groups with ``nsys profile``, including how to enable,
   disable, and target worker groups for system-level traces.

- :doc:`mbridge`
   Introduces how to use Megatron-Bridge to integrate Megatron-LM training backend,
   to support HuggingFace-format checkpoint training.

- :doc:`nsight_code_design`
   Explains the code architecture of RLinf's Nsight profiler integration and
   NVTX annotation framework -- config flow, annotation layers, and how to
   add new instrumentation.

- :doc:`nsight_profiler_guide`
   Hands-on guide to profiling RLinf with Nsight Systems.  Walks through a
   multi-node async SAC experiment and shows how to read the timeline.

.. toctree::
   :hidden:
   :maxdepth: 2

   5D
   lora
   version
   resume
   convertor
   logger
   nsight
   weight_syncer
   mbridge
   nsight_code_design
   nsight_profiler_guide
