# QNAS工程设计

`qnas/`用于在YOLOv7项目内部维护一套独立的量子线路搜索、采样筛选、TensorCircuit训练执行、Qiskit硬件分析和YOLO接入工具链。

目标是不再依赖外部`MuNA-dev_lli`目录。这里保留两条NAS路线：

```text
search_nas
  在线搜索。训练时使用结构参数alpha和量子参数theta，和YOLO loss联合优化。

sample_nas
  离线采样 + 指标筛选。先生成大量CircuitSpec，再按结构、表达能力、硬件适配性等指标选top-k，最后接入YOLO微调。
```

两条路线最终都输出同一种`CircuitSpec`，后续TensorCircuit执行、Qiskit分析、YOLO接入都只认这个统一IR。

## 目录规划

建议后续按下面结构实现：

```text
qnas/
  __init__.py

  common/
    __init__.py
    schema.py
    io.py
    gate_set.py
    registry.py

  backends/
    __init__.py
    tensorcircuit_backend.py
    qiskit_converter.py
    qiskit_metrics.py

  search_nas/
    __init__.py
    search_space.py
    searchable_layer.py
    arch_parameters.py
    export.py
    trainer_hooks.py

  sample_nas/
    __init__.py
    sampler.py
    sample_pool.py
    deduplicate.py
    rank.py
    select.py

  metrics/
    __init__.py
    structural.py
    hardware.py
    expressibility.py
    entanglement.py
    score.py

  yolo/
    __init__.py
    layers.py
    blocks.py
    factory.py

  scripts/
    sample_circuits.py
    rank_circuits.py
    export_search_arch.py
    inspect_circuit.py

  README.md
```

职责划分：

```text
common/schema.py
  定义线路结构数据类，例如GateSpec、CircuitSpec。

common/io.py
  负责CircuitSpec和json文件之间的保存、读取、校验。

common/gate_set.py
  维护支持的量子门、参数个数、是否为双比特门、硬件代价等元信息。

backends/tensorcircuit_backend.py
  训练和可微仿真后端。把CircuitSpec转换成TensorCircuit执行。

backends/qiskit_converter.py
  把CircuitSpec转换成Qiskit QuantumCircuit，用于画图、导出、transpile和硬件分析。

backends/qiskit_metrics.py
  基于Qiskit统计depth、size、count_ops、transpiled depth等指标。

search_nas/
  负责在线NAS：搜索空间、alpha/theta参数、结构参数提取、导出CircuitSpec。

sample_nas/
  负责离线采样筛选：采样候选线路、去重、计算指标、排序、保存top-k。

metrics/
  负责计算候选线路指标，例如表达能力、纠缠能力、线路深度、硬件代价。

yolo/
  负责YOLO侧的QNN层、QNN-CNN混合块和factory。它不做采样、搜索和筛选。

scripts/
  提供命令行入口，方便批量采样、筛选和检查线路。
```

## 线路文件格式

不要沿用MuNA的多文件格式：

```text
gates.txt
gate_params.txt
inputs_bounds.txt
weights_bounds.txt
meas_qubits.txt
```

建议一个线路一个json文件：

```json
{
  "name": "circ_000123",
  "n_qubits": 4,
  "n_inputs": 4,
  "n_weights": 12,
  "measured_qubits": [0, 1, 2, 3],
  "gates": [
    {"op": "ry", "wires": [0], "source": "input", "index": 0},
    {"op": "rx", "wires": [0], "source": "weight", "index": 0},
    {"op": "crx", "wires": [0, 1], "source": "weight", "index": 1},
    {"op": "cz", "wires": [1, 2], "source": "none"}
  ],
  "meta": {
    "depth": 3,
    "sampler": "layered_random_v1",
    "created_at": "2026-05-15"
  }
}
```

字段说明：

```text
name             线路名称，通常和文件名一致。
n_qubits         量子比特数。
n_inputs         从YOLO feature投影到量子线路的输入维度。
n_weights        线路内部可训练参数数量。
measured_qubits  输出测量哪些qubit的Pauli-Z期望值。
gates            按顺序执行的量子门列表。
meta             采样器、深度、实验标记等非执行信息。
```

每个gate使用统一结构：

```json
{"op": "rx", "wires": [0], "source": "weight", "index": 0}
```

```text
op       门名称，例如rx、ry、rz、cx、cz、crx、cry、crz。
wires    作用qubit。
source   参数来源：input、weight、none。
index    参数索引；source=none时可以省略。
```

## 实验目录

候选线路、指标和筛选结果建议存到`experiments/qnas/`：

```text
experiments/qnas/
  yolo_q4_l3_v1/
    config.yaml
    candidates/
      circ_000001.json
      circ_000002.json
      circ_000003.json
    metrics.csv
    best/
      top_000.json
      top_001.json
      top_002.json
    summary.json
```

`metrics.csv`建议包含：

```text
name,n_qubits,n_gates,n_1q_gates,n_2q_gates,depth,hardware_cost,expressibility,entanglement,score,path
```

## 采样策略

第一版建议实现三种采样器。

```text
layered_random
  每层每个qubit随机采样一个单比特门。
  每层按linear或ring连接模式采样双比特门。
  适合作为最小可用版本。

hardware_aware
  根据coupling_map采样双比特门。
  根据edge_error或gate_cost降低高噪声边的采样概率。
  适合后续硬件适配实验。

nas_seeded
  从NAS导出的arch出发，随机替换少量门或边。
  适合把在线NAS结果扩展成离线候选池。
```

采样接口建议：

```python
sample_layered_circuit(
    n_qubits=4,
    n_layers=3,
    oneq_ops=["rx", "ry", "rz"],
    twoq_ops=["cx", "cz", "crx", "cry", "crz"],
    entangle_pattern="linear",
    input_encoding="ry",
)
```

## 指标设计

先实现便宜、稳定、容易调试的指标。

```text
depth
  线路深度，越小越好。

n_2q_gates
  双比特门数量，越少越硬件友好。

hardware_cost
  根据gate_cost、coupling_map、edge_error加权，越低越好。

expressibility_proxy
  随机输入和随机参数下，统计测量输出的方差、熵或覆盖度，越高越好。

entanglement_proxy
  随机参数下获取state，计算单比特reduced purity或Meyer-Wallach proxy，越高越好。
```

第一版总分可以写成：

```python
score = (
    + 0.40 * expressibility_score
    + 0.20 * entanglement_score
    - 0.20 * hardware_cost_score
    - 0.10 * depth_score
    - 0.10 * twoq_gate_score
)
```

权重必须放到配置文件，不要写死在代码里。

## 后端分工

本项目不直接把`tc.Circuit`或`qiskit.QuantumCircuit`当成工程线路格式，而是使用`CircuitSpec`作为统一IR。

```text
CircuitSpec / GateSpec
  项目内部线路IR。负责保存、采样、筛选、去重、跨进程传递和YOLO接入。

TensorCircuit
  训练时执行后端。负责可微forward、state、expectation。

Qiskit QuantumCircuit
  硬件分析和导出后端。负责transpile、depth、gate_count、QASM、画图、真实设备适配。
```

数据流：

```text
search_nas -> CircuitSpec -> TensorCircuit训练 / Qiskit分析
sample_nas -> CircuitSpec -> TensorCircuit训练 / Qiskit分析
```

## YOLO接入方式

`nets/yolov7_quantum_search.py`不应该知道采样和筛选细节，只接收最终线路文件或top-k目录。

建议统一模式：

```text
quantum_mode="search"
  使用在线search NAS层，有alpha结构参数。

quantum_mode="sampled"
  使用离线筛选出的固定CircuitSpec。

quantum_mode="none"
  不插入量子层，用于shape检查和debug。
```

单线路配置：

```yaml
model:
  module: nets.yolov7_quantum_search
  kwargs:
    quantum_mode: sampled
    quantum_positions: [p5_spp]
    circuit_path: experiments/qnas/yolo_q4_l3_v1/best/top_000.json
```

top-k目录配置：

```yaml
model:
  module: nets.yolov7_quantum_search
  kwargs:
    quantum_mode: sampled
    quantum_positions: [p5_spp]
    circuit_dir: experiments/qnas/yolo_q4_l3_v1/best
    circuit_rank: 0
```

## 命令行入口

Search NAS最小拟合测试：

```bash
python qnas/examples/toy_search_fit.py \
  --backend torch \
  --n-qubits 4 \
  --n-layers 2 \
  --num-samples 128 \
  --batch-size 32 \
  --steps 80 \
  --out experiments/qnas/toy_search/best_arch.json
```

正式 TensorCircuit 后端测试：

```bash
python qnas/examples/toy_search_fit.py \
  --backend tensorcircuit \
  --tc-backend pytorch \
  --n-qubits 4 \
  --n-layers 2 \
  --num-samples 128 \
  --batch-size 32 \
  --steps 80 \
  --out experiments/qnas/toy_search_tc/best_arch.json
```

`--backend tensorcircuit --tc-backend pytorch`会用TensorCircuit的PyTorch backend执行量子态演化，`alpha`和`theta`仍然是PyTorch参数，不需要TensorFlow。

如果要复现旧`llin_qnn`里的 TensorFlow backend + `torch_interface` 路线，可以改成：

```bash
python qnas/examples/toy_search_fit.py \
  --backend tensorcircuit \
  --tc-backend tensorflow \
  --n-qubits 4 \
  --n-layers 2 \
  --num-samples 128 \
  --batch-size 32 \
  --steps 80 \
  --out experiments/qnas/toy_search_tc_tf/best_arch.json
```

该模式需要当前环境额外安装`tensorflow`。

这个脚本使用随机生成的回归数据验证：

```text
SearchableQuantumLayer可以forward
alpha结构参数和theta量子参数可以优化
loss可以下降
搜索出的结构可以导出为CircuitSpec json
```

输出的json是一个完整search结果，包含：

```text
circuit
  导出的离散CircuitSpec结构。

parameters
  alpha_1q / alpha_2q
  alpha_1q_softmax / alpha_2q_softmax
  theta_1q / theta_2q
  toy head_weight / head_bias

training
  backend、seed、steps、learning rate、initial_loss、final_loss、loss_ratio等摘要。
```

采样候选线路：

```bash
python qnas/scripts/sample_circuits.py \
  --config configs/qnas_sample_yolo_q4_l3.yaml \
  --out experiments/qnas/yolo_q4_l3_v1 \
  --num-circuits 1000
```

计算指标并筛选：

```bash
python qnas/scripts/rank_circuits.py \
  --config configs/qnas_rank_yolo_q4_l3.yaml \
  --circuits experiments/qnas/yolo_q4_l3_v1/candidates \
  --out experiments/qnas/yolo_q4_l3_v1 \
  --top-k 25
```

训练YOLO：

```bash
python train_configurable_model.py \
  --config configs/yolo_quantum_sampled.yaml
```

## 推荐实现顺序

```text
1. 实现CircuitSpec / GateSpec和json读写。
2. 实现TensorCircuit runner，支持rx、ry、rz、cx、cz、crx、cry、crz。
3. 实现Qiskit converter，用于硬件分析和导出。
4. 实现sample_circuits.py，能生成候选json。
5. 实现depth、n_2q_gates、hardware_cost三个无仿真指标。
6. 实现rank_circuits.py，输出metrics.csv和best/top_*.json。
7. 实现search_nas的SearchableQuantumLayer可微forward。
8. 实现expressibility_proxy和entanglement_proxy。
9. 在nets侧接入qnas.yolo.factory。
```

## 协作约定

```text
qnas/common/
  只处理线路IR、读写、门集合，不引入YOLO模型。

qnas/backends/
  只处理CircuitSpec到具体后端的转换和执行。

qnas/search_nas/
  只处理在线搜索NAS，不处理离线候选池排序。

qnas/sample_nas/
  只处理离线采样和筛选，不处理YOLO模型骨架。

qnas/metrics/
  只处理指标，不直接保存best线路。

qnas/yolo/
  负责把CircuitSpec接到CNN feature map。

nets/
  只保留最终模型结构和MODEL_CLASS，不放采样和筛选逻辑。
```

新增量子门时，需要同步更新：

```text
qnas/common/gate_set.py
qnas/backends/tensorcircuit_backend.py
qnas/backends/qiskit_converter.py
qnas/metrics/hardware.py
```
