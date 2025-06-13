# 双线偏压地表沉降计算程序

这个程序用于计算双线偏压条件下的地表沉降函数 W(x)。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

运行主程序：

```bash
python tunnel_settlement.py
```

程序会提示输入以下参数：
1. 隧道埋深 (H)
2. 隧道半径 (R)
3. 地层损失率 (Vl)
4. 第一偏压参数 (u1)
5. 第三偏压参数 (u3)
6. 偏压角度 (theta)
7. 平均内摩擦角 (fai)
8. 土质参数 (k)

输入参数后，程序将计算 W(x) 函数并绘制图形。

## 输出说明

程序将生成一个图形，显示 W(x) 函数随 x 的变化关系。图形将保存为 'settlement_curve.png'。 