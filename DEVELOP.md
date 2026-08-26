# 开发文档

## 安装

```bash
# 注意要用虚拟环境，不要覆盖主机的原版 deep_ep
source env3.12/bin/activate

rm -rf build dist *.egg-info

# nvshmem 已随 paddle 自动安装, 无需指定路径
python setup.py bdist_wheel

python -m pip install --force-reinstall dist/deep_ep-*.whl
```


## 单测

使用 mpirun 在集群的 node 0 和 node 1 上运行

当前集群为共享开发机，性能会有波动，当前开发阶段以正确性为主

```bash
source env3.12/bin/activate
cd tests_overlap
mpirun python run.py 0,1 test_internode_baseline.py 
```


## 开发进展

8.26:

dispatch + unzip 融合的第一步（不带信号）已完成，在 internode dispatch 的 NVL receiver 上加多播。

- 实现：`csrc/kernels/internode.cu` 的 `kNVLReceivers` 分支，token 的 TMA 数据已在 smem，多播只是对每个命中的本地专家再发一次 `tma_store_1d`；槽位用 `atomicAdd` 抢，专家段基址是 CPU 侧算好的对齐前缀和
- 开关：`buffer.dispatch(..., unzip_alignment=128)`，kernel 内部靠 `unzipped_x` 指针是否为空判断，接口复用，`unzip_alignment=0` 时行为与原版完全一致
- 单测：`tests_overlap/test_internode_fused_unzip.py`，把 token 全局 id 编码进 x 的前三列，按 id 排序后与 `moe_permute` 做逐行精确比对（因为段内是 atomic 序，只能做集合比对）
- 正确性：PASS
- 性能（SEQLEN=16384, H=4096, TOPK=8, E=16, 16 ranks, 48 SM）：
  - dispatch only: 1.96 ms
  - dispatch + moe_permute: 2.87 ms
  - dispatch 融合 unzip: 2.44 ms

即融合本身给 dispatch 增加了约 0.48 ms（+24%），但比现在 dispatch + 独立 unzip 的两段式快约 15%。

待办：padding 行目前是未初始化的脏数据（计算侧不能统计这些行）；`atomic_to_zip` / `task_queue` 等信号还没做；zip 还需要一张 atomic 序的反向表才能定位一个 token 的各专家输出。
