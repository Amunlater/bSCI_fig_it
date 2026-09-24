# figIndividual — 按统计图表类型拆分的科研绘图工具箱

把论文 Fig3–Fig10 的全部绘图代码，**按统计图表类别**拆分为独立模块：一个 `.py`
只生成**一种类型**的图。配套一个**统一入口**（识别输入数据）和**统一输出端口**
（图表 / 结果表格 / 处理后数据 / 报告）。

---

## 1. 目录结构

```
figIndividual/
├── run_individual.py   # ★ 统一入口（命令行）
├── example.py          # 随机数据演示
├── pipeline.py         # 编排：识别 -> 加载 -> 选图 -> 输出
├── dataset_loader.py   # 识别 GEO/TCGA/表达矩阵/临床/注释/原始测序
├── sc_loader.py        # 单细胞原始计数读取（10x / *expression*.gz）
├── sc_pipeline.py      # 单细胞分析流程（QC->PCA->UMAP->注释->sc_* 字段）
├── registry.py         # 图表类型注册表
├── config.py           # 配色、matplotlib 设置、默认输出目录
├── data_interface.py   # PaperData 数据契约
├── simulate.py         # 随机/论文统计量数据生成（demo 与缺失字段回填）
├── io_utils.py         # 统一输出端口 OutputManager
├── requirements.txt    # 第三方依赖清单
├── charts/             # ★ 图表模块：一个 .py 只生成一种图表类型
│   ├── __init__.py
│   ├── chart_survival.py   # KM 生存曲线
│   ├── chart_roc.py        # ROC 曲线
│   ├── chart_forest.py     # 森林图（Cox）
│   ├── chart_lasso.py      # LASSO 路径
│   ├── chart_calibration.py# 校准曲线
│   ├── chart_dca.py        # 决策曲线（DCA）
│   ├── chart_nomogram.py   # 列线图
│   ├── chart_dimred.py     # PCA / t-SNE / UMAP 降维
│   ├── chart_heatmap.py    # 表达热图 / 配体-受体热图
│   ├── chart_bar.py        # 条形图（AUC、GSEA、免疫浸润、空间统计…）
│   ├── chart_box.py        # 箱线图（ESTIMATE、COMP、CD8 密度）
│   ├── chart_violin.py     # 小提琴图（衰老、药物 IC50）
│   ├── chart_scatter.py    # 散点/空间分布
│   ├── chart_network.py    # DE-CSRG 网络图
│   ├── chart_image.py      # 多重免疫荧光 / IHC 图像
│   └── chart_table.py      # 结果汇总表
└── readme.md
```

---

## 2. 快速开始

```powershell
# 进入目录
cd F:\RProject\figures\figIndividual

# ① 随机数据演示（无需任何数据）
python example.py

# ② 自动识别数据集并自动选择能画的图
python run_individual.py "F:\RProject\GSE\GSE222315_RAW" -o .\output

# ③ 只画指定类型
python run_individual.py "D:\data\TCGA_BLCA" -o .\output --charts survival,roc,bar

# ④ 画全部类型（缺失字段用模拟值回填）
python run_individual.py "D:\data\GSE39582" -o .\output --charts all

# ⑤ 查看所有图表类型
python run_individual.py --list-charts
```

> Windows PowerShell 下路径建议用引号包裹。

---

## 3. 命令行参数（`run_individual.py`）

| 参数 | 说明 | 默认 |
|---|---|---|
| `input`（位置参数） | 输入文件路径（含数据集的文件夹） | 省略则用模拟数据 |
| `-o, --output` | 输出根目录 | `figIndividual/output` |
| `-c, --charts` | 图表类型，逗号分隔；`all` 表示全部 | 自动选择 |
| `--demo` | 强制使用模拟数据（忽略 input） | 否 |
| `--n-cap` | 单细胞每样本最大抽样细胞数 | `2500` |
| `--seed` | 随机种子 | `42` |
| `--force` | 重建单细胞缓存 | 否 |
| `--no-fill` | 不用模拟值回填缺失字段 | 否 |
| `--format` | 图片格式 `png`/`pdf`/`svg` | `png` |
| `--list-charts` | 列出所有图表类型与所需字段 | — |

---

## 4. 支持的图表类型

| 类型 | 内容 | 所需 PaperData 字段 |
|---|---|---|
| `survival` | Kaplan-Meier 生存曲线 | `survival_times/events/risk_groups` |
| `roc` | 时间依赖 ROC + 队列 ROC | `roc_1yr/3yr/5yr` |
| `forest` | 单/多因素 Cox 森林图 | `gene_cox` |
| `lasso` | LASSO 系数路径 + CV | `gene_cox` |
| `calibration` | 校准曲线 | `calibration_curves` |
| `dca` | 决策曲线分析 | `c_index_data` |
| `nomogram` | 列线图 | `c_index_data` |
| `dimred` | PCA / t-SNE / UMAP | `pca_all` |
| `heatmap` | 表达热图 / CellChat 热图 | `gene_expr`,`risk_scores` |
| `bar` | AUC/GSEA/免疫浸润/空间统计等 | `gsea_high` |
| `box` | ESTIMATE / COMP / CD8 箱线图 | `estimate_scores` |
| `violin` | 衰老评分 / 药物 IC50 | `drug_ic50` |
| `scatter` | 风险评分 / 空间分布 / 距离衰减 | `risk_scores` |
| `network` | DE-CSRG 基因网络 | `gene_cox` |
| `image` | mIF / IHC 图像 | `cd8_density` |
| `table` | 结果汇总表 | `comp_roc` |

各字段含义见 `data_interface.py:PaperData`。

---

## 5. 输入数据识别

`dataset_loader.py` 递归扫描输入文件夹并按文件名归类：

| 识别角色 | 举例文件名 | 处理方式 |
|---|---|---|
| 原始单细胞 | `matrix.mtx` / `barcodes.tsv` / `features.tsv` / `*expression*.gz` | 内置流程 `sc_loader.py`+`sc_pipeline.py` |
| GEO series matrix | `*series_matrix*.txt` | 解析分组/表达 |
| 表达矩阵 | `*counts* / *fpkm* / *tpm* / *.csv / *.tsv / *.xlsx` | 自动判断基因×样本方向 |
| 样本信息/分组 | `*clinical* / *survival* / *phenotype* / *metadata* / *samples*` | 提取生存时间/事件/分组 |
| 注释文件 | `*GPL* / *platform* / *annot* / *.gtf* / *.gff*` | 识别并记录（用于基因映射） |
| 原始测序数据 | `*.fastq* / *.bam / *.sra` | 无定量 → 记录并回退模拟 |

判定结果写入报告 `reports/summary_report.md`。数据能真实推导的字段（风险分、分组、
签基因表达、PCA、生存）会被真实计算；其余字段在需要时由 `simulate.py` 回填。
若想严格只用真实数据，加 `--no-fill`。

---

## 6. 统一输出端口

所有结果写入 `--output` 指定的根目录：

```
output/
├── figures/      # <类型>.png          每类图表
├── tables/       # <类型>__<表名>.csv  图中数字
├── processed/    # <类型>__<名>.csv/npy/json  处理后数据
├── reports/      # <类型>.md           单图说明
│                 # summary_report.md   本次运行总报告
└── manifest.csv  # 输出清单
```

---

## 7. 作为库调用

```python
import sys; sys.path.insert(0, r"F:\RProject\figures\figIndividual")
from pipeline import run, available_charts
from simulate import simulate_paper_data
from data_interface import PaperData

# 方式一：直接用随机数据跑全部图
run(demo=True, out_dir=r"F:\RProject\figures\figIndividual\output")

# 方式二：自己填数据后只画两张
d = PaperData()
d.risk_scores = ...; d.risk_groups = ...; d.gene_expr = ...
run(out_dir="out", charts=["survival", "heatmap"])  # 或手动调用 chart 模块
```

单图模块接口统一为：

```python
from charts import chart_forest
out = chart_forest.build(data)     # 返回 ChartOutput(figure, tables, processed, report)
```

---

## 8. 新增图表类型

1. 新建 `charts/chart_<name>.py`，定义 `CHART_TYPE / TITLE / REQUIRED` 和
   `build(data) -> ChartOutput`；
2. 在 `registry.py` 的 `CHART_MODULES` 列表末尾加上模块名
   `"charts.chart_<name>"`。

无需改动入口与输出代码。
