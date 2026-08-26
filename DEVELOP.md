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
