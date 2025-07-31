# Proactive Fault Tolerance System for Megatron-LM Training

A comprehensive fault tolerance framework that enhances NVIDIA Megatron-LM with proactive GPU failure detection, emergency checkpointing, and automated recovery capabilities for large-scale language model training.

## Overview

Large Language Model (LLM) training on distributed GPU clusters is vulnerable to hardware failures that can result in significant computational loss. This system addresses these challenges by providing:

- **Real-time GPU monitoring** with anomaly detection
- **Proactive emergency checkpointing** triggered by hardware health indicators
- **Automated recovery** with dynamic parallelism reconfiguration
- **Minimal training disruption** through asynchronous operations

## Key Features

### 🔍 GPU Health Monitoring
- Continuous monitoring of critical GPU metrics (temperature, memory, power, utilization)
- Dynamic risk score computation using weighted anomaly detection
- Real-time alerting system for potential hardware failures

### ⚡ Emergency Checkpointing
- Proactive checkpoint triggering based on GPU health indicators
- Smart filtering to prevent redundant checkpoints (configurable gap thresholds)
- Integration with Megatron-LM's asynchronous checkpointing for minimal overhead

### 🔄 Automated Recovery
- Seamless training resumption after GPU failures
- Dynamic recalculation of tensor and pipeline parallelism configurations
- Support for training continuation with reduced GPU resources

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Master Node   │    │ GPU Monitoring  │    │ Training Process│
│                 │    │                 │    │                 │
│ • Orchestrates  │◄──►│ • Metrics       │◄──►│ • Megatron-LM   │
│   all processes │    │   Collection    │    │   Training      │
│ • Manages       │    │ • Risk Score    │    │ • Emergency     │
│   recovery      │    │   Computation   │    │   Checkpointing │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### GPU Monitoring System

The monitoring subsystem tracks critical GPU metrics and computes a composite risk score:

```python
# Risk Score Computation
Risk_Score = Σ(w_i × r_i)

where:
- w_i = weight for metric i
- r_i = normalized risk factor for metric i

weights = {
    'temperature': 0.25,
    'memory': 0.15,
    'errors': 0.25,
    'power': 0.10,
    'utilization_drop': 0.15,
    'temp_rise': 0.10
}
```

**Monitored Metrics:**
- GPU temperature and thermal trends
- Memory utilization and available headroom
- Power draw fluctuations
- GPU utilization patterns
- Hardware error indicators

### Emergency Checkpoint Strategy

```python
def should_emergency_save(iteration, last_checkpoint_iter, save_interval, threshold_frac=0.25):
    if not os.path.exists('/workspace/megatron2/save_now.flag'):
        return False
    gap = iteration - last_checkpoint_iter
    return gap >= int(threshold_frac * save_interval)

# Modified checkpoint condition
if args.save and (
    iteration % args.save_interval == 0 or
    should_emergency_save(iteration, last_checkpoint_iter, args.save_interval)):
    save_checkpoint(...)
    last_checkpoint_iter = iteration
```

### Automated Recovery Logic

The recovery mechanism dynamically reconfigures parallelism based on available resources:

```python
# Parallelism Reconfiguration
TP × PP = Number_of_Available_GPUs
DP = Number_of_Available_GPUs / (TP × PP)

# Constraints:
# 1. Number_of_Attention_Heads mod TP = 0
# 2. Number_of_Available_GPUs mod PP = 0
# 3. TP × PP × DP = Number_of_Available_GPUs
```


## Configuration

### GPU Monitoring Parameters
```python
# Monitoring intervals and thresholds
MONITORING_INTERVAL = 10  # seconds
RISK_THRESHOLD = 0.95     # trigger emergency checkpoint
ROLLING_WINDOW_SIZE = 60  # samples for baseline calculation
```

### Emergency Checkpoint Settings
```python
# Checkpoint gap filtering
CHECKPOINT_GAP_THRESHOLD = 0.25  # 25% of save_interval
EMERGENCY_CHECK_INTERVAL = 30    # seconds
```

## Performance Analysis

### Checkpoint Overhead (Worst-Case Scenario)

| Model Size | Baseline Time | 50% Gap | 25% Gap |
|------------|---------------|---------|---------|
| 857M       | 476s         | 952s    | 1904s   |
| 1.7B       | 1624s        | 3248s   | 6496s   |
| 7.1B       | 7358s        | 14716s  | 29432s  |

*Note: With asynchronous checkpointing enabled, actual overhead is significantly reduced as checkpoints run in background.*

## File Structure

```
Megatron-LM/
├── gpu_metric/                        # GPU metric collection and failure simulation
│   ├── gpu_logs/                      # Raw logs directory
│   ├── gpu_detailed_metrics.csv       # Detailed GPU usage data
│   ├── gpu_detailed_metrics_intenseload.csv  # Metrics under high-load scenarios
│   ├── gpu_metrics.csv                # Aggregated GPU metrics
│   ├── checkpointing.py               # Checkpoint save/load utilities
│   ├── gpu_failure_intenseload.py     # Simulate failures under intense GPU usage
│   ├── gpu_failure_simulation.py      # General-purpose GPU failure simulation
│   ├── metric_collection.py           # Scripts to collect GPU health and usage data
│   └── recover.py                     # Logic for recovery after GPU faults
│
├── master_scripts/                    # Orchestration and control
│   ├── master_node.py                 # Central controller for scheduling and coordination
│   ├── test.py                        # Integration test entry point
│   └── recovery_utils.py              # Shared recovery helper functions
│
└── megatron/                          # Megatron-LM training scripts and examples
    ├── training/                      # Core training implementation
    │   └── training.py                # Launch training with fault tolerance hooks
    ├── examples/gpt3/                 # Example GPT-3 configs and helper files
    ├── train_gpt3_175b_distributed.sh # Shell script to train GPT-3 175B model
    └── README.md                      # Megatron-specific documentation
```

## Key Innovations

1. **Proactive Failure Detection**: Unlike traditional reactive approaches, our system predicts potential failures before they occur
2. **Smart Checkpoint Filtering**: Prevents checkpoint spam while ensuring critical saves occur
3. **Dynamic Parallelism Reconfiguration**: Automatically adjusts training configuration based on available resources
4. **Minimal Training Disruption**: Leverages asynchronous operations to maintain training throughput

## Supported Models

- GPT-3 (345M, 857M, 1.7B, 7.1B+ parameters)
- BERT variants
- Mixtral architectures
- Custom transformer models

## System Requirements

- **Minimum**: 4x NVIDIA V100 GPUs (32GB each)
- **Recommended**: 8x NVIDIA A100 GPUs (80GB each)
- **Storage**: High-performance file system (recommended: VAST, Lustre)
- **Network**: High-bandwidth interconnect (InfiniBand recommended)

## Limitations & Future Work

- GPU failure prediction models require hardware-specific calibration
- Limited by availability of public GPU failure datasets
- Storage requirements scale with checkpoint frequency
- Future enhancements could include cross-framework compatibility

## Research Paper

This implementation is based on the research paper: **"Proactive Fault Tolerance System for Megatron-LM Training"**. 👉 Please refer to the Proactive Fault Tolerance System for Megatron LMTraining.pdf file in this repository for the full PDF of the paper and detailed methodology.


---

**For detailed usage instructions and advanced configuration options, please refer to the original Megatron-LM documentation and our research paper.**
