# ============================================================
# 桩基低应变完整性智能检测系统
# 多分类 / 二分类（CFG桩、空心方桩）双模式部署版
#
# 模型路由：
# 1) 多分类：
#    cnn_weights_only.weights.h5
#    best_tabpfn_model.pkl
#    super_scaler.pkl
#
# 2) 二分类 - CFG桩：
#    04 CFG桩张家场回迁区_cnn_model.h5
#    04 CFG桩张家场回迁区_tabpfn_model.pkl
#    04 CFG桩张家场回迁区_super_scaler.pkl
#
# 3) 二分类 - 空心方桩：
#    06空心方桩_cnn_model.h5
#    06空心方桩_tabpfn_model.pkl
#    06空心方桩_super_scaler.pkl
#
# 重要：
# 二分类模型文件只能确认数值标签为 0 / 1，文件本身不保存
# “0=完整、1=缺陷”的文字语义。下面暂按这一常用映射显示。
# 若你的训练标签相反，只需交换 BINARY_DIAGNOSIS_MAP 两行文字。
# ============================================================

import os
from pathlib import Path

# Streamlit Community Cloud 为 CPU 环境：
# 在任何 TensorFlow 导入之前彻底禁用 GPU/CUDA 探测，避免 cuInit 崩溃。
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 尽量避免云端瞬时线程过多
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import streamlit as st
import numpy as np

# 重型/可选依赖全部延迟导入，避免应用启动阶段直接崩溃
try:
    import pandas as pd
except Exception as _pd_error:
    st.error(f"pandas 导入失败：{type(_pd_error).__name__}: {_pd_error}")
    st.stop()


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="桩基低应变智能诊断系统",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ============================================================
# 类别映射
# ============================================================
MULTI_DIAGNOSIS_MAP = {
    0: "完整桩 (Ⅰ类)",
    1: "局部缩径桩 (Ⅱ类)",
    2: "桩头破碎/差 (Ⅲ类)",
    3: "严重断裂桩 (Ⅳ类)",
    4: "局部扩径桩 (反相)"
}

# 如你的二分类训练标签相反，只交换下面两行文字即可。
BINARY_DIAGNOSIS_MAP = {
    0: "完整桩",
    1: "缺陷/异常桩"
}

PHYS_COLS = [
    "Diameter", "Length", "Velocity", "RMS", "Total_Energy",
    "Peak_to_Peak", "Abs_Mean", "Kurtosis", "Skewness",
    "Std_Dev", "Shape_Factor", "Crest_Factor", "Impulse_Factor",
    "Clearance_Factor", "CV"
]

FEATURE_NAMES_23D = [
    "CNN概率(断裂)", "CNN概率(扩径)", "CNN概率(缩径)",
    "CNN概率(完整)", "CNN概率(桩头)",
    "桩径(Diameter)", "桩长(Length)", "波速(Velocity)",
    "RMS", "总能量", "峰峰值", "绝对均值",
    "峰度", "偏度", "标准差", "波形因数",
    "峰值因数", "脉冲因数", "裕度因数", "变异系数",
    "动态波形偏度", "动态波形峰度", "FFT主频能量比"
]

# 二分类 scaler 已核对为 17 维：
# 1 个 CNN sigmoid 输出 + 15 个现场/统计特征 + FFT 主频能量比
FEATURE_NAMES_17D = [
    "CNN输出概率(类别1)",
    "桩径(Diameter)", "桩长(Length)", "波速(Velocity)",
    "RMS", "总能量", "峰峰值", "绝对均值",
    "峰度", "偏度", "标准差", "波形因数",
    "峰值因数", "脉冲因数", "裕度因数", "变异系数",
    "FFT主频能量比"
]


# ============================================================
# 模型文件配置
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_CONFIG = {
    "多分类模式": {
        "cnn": [
            "cnn_weights_only.weights.h5",
            "cnn_weights_only.weights(1).h5"
        ],
        "scaler": [
            "super_scaler.pkl",
            "super_scaler(2).pkl",
            "super_scaler(1).pkl"
        ],
        "tabpfn": [
            "best_tabpfn_model.pkl",
            "best_tabpfn_model(2).pkl",
            "best_tabpfn_model(1).pkl"
        ],
        "cnn_outputs": 5,
        "n_features": 23,
        "feature_names": FEATURE_NAMES_23D
    },
    "二分类模式": {
        "水泥粉煤灰碎石桩（CFG桩）": {
            "cnn": ["04 CFG桩张家场回迁区_cnn_model.h5"],
            "scaler": ["04 CFG桩张家场回迁区_super_scaler.pkl"],
            "tabpfn": ["04 CFG桩张家场回迁区_tabpfn_model.pkl"],
            "cnn_outputs": 1,
            "n_features": 17,
            "feature_names": FEATURE_NAMES_17D
        },
        "空心方桩": {
            "cnn": ["06空心方桩_cnn_model.h5"],
            "scaler": ["06空心方桩_super_scaler.pkl"],
            "tabpfn": ["06空心方桩_tabpfn_model.pkl"],
            "cnn_outputs": 1,
            "n_features": 17,
            "feature_names": FEATURE_NAMES_17D
        }
    }
}


def first_existing(candidates):
    """从候选文件名中找到第一个存在的文件。"""
    for name in candidates:
        p = BASE_DIR / name
        if p.exists():
            return p
    return None


def build_multiclass_cnn_from_weights(tf):
    """
    根据多分类 cnn_weights_only.weights.h5 的真实权重形状恢复 CNN 骨架。

    已核对的主要权重形状：
    Conv1D-1: (5, 1, 32)
    BatchNormalization: 4 x (32,)
    Conv1D-2: (3, 32, 64)
    Dense-1: (64, 32)
    Dense-2: (32, 5)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(256, 1), name="wave_input"),
        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=5,
            activation="relu",
            name="conv1d"
        ),
        tf.keras.layers.BatchNormalization(
            name="batch_normalization"
        ),
        tf.keras.layers.MaxPooling1D(
            pool_size=2,
            name="max_pooling1d"
        ),
        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            name="conv1d_1"
        ),
        tf.keras.layers.GlobalAveragePooling1D(
            name="global_average_pooling1d"
        ),
        tf.keras.layers.Dense(
            units=32,
            activation="relu",
            name="dense"
        ),
        tf.keras.layers.Dropout(
            rate=0.3,
            name="dropout"
        ),
        tf.keras.layers.Dense(
            units=5,
            activation="softmax",
            name="dense_1"
        )
    ])
    return model


def selected_config(diagnosis_mode, pile_type=None):
    if diagnosis_mode == "多分类模式":
        return MODEL_CONFIG["多分类模式"]
    return MODEL_CONFIG["二分类模式"][pile_type]


# ============================================================
# 按需加载模型
# 只缓存最近选择的一组，避免三套 TabPFN 同时长期占用内存
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model_bundle(diagnosis_mode, pile_type_key):
    import joblib
    import tensorflow as tf

    # 限制 TF 线程，降低 Community Cloud 瞬时 CPU 压力
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass

    pile_type = None if pile_type_key == "__MULTI__" else pile_type_key
    cfg = selected_config(diagnosis_mode, pile_type)

    cnn_path = first_existing(cfg["cnn"])
    scaler_path = first_existing(cfg["scaler"])
    tabpfn_path = first_existing(cfg["tabpfn"])

    missing = []
    if cnn_path is None:
        missing.append("CNN: " + " / ".join(cfg["cnn"]))
    if scaler_path is None:
        missing.append("Scaler: " + " / ".join(cfg["scaler"]))
    if tabpfn_path is None:
        missing.append("TabPFN: " + " / ".join(cfg["tabpfn"]))

    if missing:
        raise FileNotFoundError(
            "当前模式缺少模型文件：\n" + "\n".join(missing)
        )

    # CNN 加载方式不同：
    # - 多分类提供的是 weights-only 文件，需要先恢复骨架再 load_weights
    # - 两个二分类提供的是完整 .h5 模型，可直接 load_model
    if diagnosis_mode == "多分类模式":
        cnn = build_multiclass_cnn_from_weights(tf)
        _ = cnn(
            np.zeros((1, 256, 1), dtype=np.float32),
            training=False
        )
        cnn.load_weights(str(cnn_path))
    else:
        cnn = tf.keras.models.load_model(
            str(cnn_path),
            compile=False
        )

    scaler = joblib.load(scaler_path)
    tabpfn = joblib.load(tabpfn_path)

    # ---------- 结构一致性检查 ----------
    input_shape = tuple(cnn.input_shape[1:])
    if input_shape != (256, 1):
        raise ValueError(
            f"CNN 输入维度不匹配：得到 {cnn.input_shape}，预期 (None, 256, 1)"
        )

    actual_cnn_outputs = int(cnn.output_shape[-1])
    if actual_cnn_outputs != cfg["cnn_outputs"]:
        raise ValueError(
            f"CNN 输出维度不匹配：得到 {actual_cnn_outputs}，"
            f"预期 {cfg['cnn_outputs']}"
        )

    scaler_n = getattr(scaler, "n_features_in_", None)
    if scaler_n is not None and int(scaler_n) != cfg["n_features"]:
        raise ValueError(
            f"Scaler 特征数不匹配：得到 {scaler_n}，"
            f"预期 {cfg['n_features']}"
        )

    return {
        "cnn": cnn,
        "scaler": scaler,
        "tabpfn": tabpfn,
        "config": cfg,
        "paths": {
            "cnn": cnn_path.name,
            "scaler": scaler_path.name,
            "tabpfn": tabpfn_path.name
        }
    }


# ============================================================
# 特征提取
# ============================================================
def extract_dynamic_features(wave_input):
    wave = np.asarray(wave_input, dtype=np.float64)

    mean = np.mean(wave)
    centered = wave - mean
    m2 = np.mean(centered ** 2)

    if m2 <= 1e-20:
        s_val = 0.0
        k_val = 0.0
    else:
        m3 = np.mean(centered ** 3)
        m4 = np.mean(centered ** 4)
        s_val = float(m3 / (m2 ** 1.5))
        k_val = float(m4 / (m2 ** 2) - 3.0)

    fft_mag = np.abs(np.fft.fft(wave))
    half_mag = fft_mag[1:128]
    f_ratio = float(
        np.max(half_mag) / (np.sum(half_mag) + 1e-8)
    )

    return s_val, k_val, f_ratio


def build_multiclass_features(cnn_probs, phys_input, s_val, k_val, f_ratio):
    """
    多分类：5 CNN 概率 + 15 phys + 3 动态特征 = 23
    """
    x = np.hstack((
        np.asarray(cnn_probs, dtype=float).reshape(-1),
        np.asarray(phys_input, dtype=float).reshape(-1),
        [s_val, k_val, f_ratio]
    ))
    if x.size != 23:
        raise ValueError(f"多分类融合特征应为23维，当前为{x.size}维。")
    return x.reshape(1, -1)


def build_binary_features(cnn_prob_class1, phys_input, f_ratio):
    """
    二分类：1 CNN sigmoid 概率 + 15 phys + 1 FFT比值 = 17
    """
    x = np.hstack((
        [float(cnn_prob_class1)],
        np.asarray(phys_input, dtype=float).reshape(-1),
        [f_ratio]
    ))
    if x.size != 17:
        raise ValueError(f"二分类融合特征应为17维，当前为{x.size}维。")
    return x.reshape(1, -1)


# ============================================================
# 预测概率与类别对齐
# ============================================================
def get_final_prediction(tabpfn_model, x_scaled):
    """
    只调用一次 predict_proba，减少 TabPFN 重复推理。
    返回：
      predicted_label: TabPFN 的真实数值类别标签
      probs: predict_proba 原始概率
      classes: 与 probs 对应的类别标签
      pred_position: 概率数组中的位置（供 SHAP 输出选择）
    """
    probs = np.asarray(
        tabpfn_model.predict_proba(x_scaled),
        dtype=float
    )[0]

    classes = np.asarray(
        getattr(tabpfn_model, "classes_", np.arange(len(probs)))
    )

    pred_position = int(np.argmax(probs))
    predicted_label = int(classes[pred_position])

    return predicted_label, probs, classes, pred_position


# ============================================================
# 绘图
# ============================================================
def render_probability_bar_chart(probs, labels, predicted_position, title):
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    probs = np.asarray(probs, dtype=float)
    pct = probs * 100.0

    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=120)

    colors = [
        "#2f6f9f" if i == predicted_position else "#d3d3d3"
        for i in range(len(labels))
    ]

    bars = ax.bar(
        labels, pct,
        color=colors,
        width=0.52,
        edgecolor="black",
        linewidth=0.7
    )

    ymax = max(100.0, float(np.max(pct)) * 1.20)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("判定置信度 (%)")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close(fig)


def calculate_shap_explanation(
    tabpfn_model,
    x_scaled,
    n_features,
    predicted_position,
    nsamples=20
):
    """
    仅在“完整解释”模式调用。
    KernelExplainer 很耗 CPU，因此默认 nsamples=20。
    """
    import shap

    background = np.zeros(
        (3, n_features),
        dtype=np.float32
    )

    explainer = shap.KernelExplainer(
        tabpfn_model.predict_proba,
        background
    )

    shap_values_all = explainer.shap_values(
        x_scaled,
        nsamples=nsamples
    )

    expected = np.asarray(
        explainer.expected_value
    ).reshape(-1)

    if isinstance(shap_values_all, list):
        shap_values = np.asarray(
            shap_values_all[predicted_position][0],
            dtype=float
        )
    else:
        arr = np.asarray(shap_values_all)
        if arr.ndim == 3:
            # 常见新版 SHAP: (samples, features, outputs)
            shap_values = np.asarray(
                arr[0, :, predicted_position],
                dtype=float
            )
        elif arr.ndim == 2:
            shap_values = np.asarray(
                arr[0, :],
                dtype=float
            )
        else:
            raise ValueError(
                f"无法识别 SHAP 输出维度：{arr.shape}"
            )

    if expected.size == 1:
        base_value = float(expected[0])
    else:
        base_value = float(expected[predicted_position])

    return base_value, shap_values


def render_waterfall_plot(
    base_value,
    shap_values,
    feature_names,
    feature_values,
    target_class_name
):
    import shap
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    shap_exp = shap.Explanation(
        values=np.asarray(shap_values, dtype=float),
        base_values=float(base_value),
        data=np.asarray(feature_values, dtype=float),
        feature_names=feature_names
    )

    plt.figure(figsize=(9, 5), dpi=120)
    shap.plots.waterfall(
        shap_exp,
        max_display=min(10, len(feature_names)),
        show=False
    )
    plt.title(
        f"【{target_class_name}】局部决策特征归因",
        fontsize=12,
        fontweight="bold"
    )
    plt.tight_layout()
    st.pyplot(plt.gcf(), width='stretch')
    plt.close()


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🎛️ 控制面板")
    st.markdown("---")

    st.markdown("### 🧠 1. 诊断模式")
    diagnosis_mode = st.radio(
        "选择分类任务",
        ["多分类模式", "二分类模式"],
        index=0,
        horizontal=False
    )

    pile_type = None
    if diagnosis_mode == "二分类模式":
        pile_type = st.selectbox(
            "选择检测桩型",
            [
                "水泥粉煤灰碎石桩（CFG桩）",
                "空心方桩"
            ]
        )

    st.markdown("### ⚡ 2. 推理方式")
    inference_mode = st.radio(
        "选择运行方式",
        [
            "快速诊断（CNN + TabPFN）",
            "完整解释（CNN + TabPFN + SHAP）"
        ],
        index=0,
        help=(
            "快速诊断不计算SHAP，诊断模型本身不降级；"
            "完整解释会额外进行SHAP计算，因此明显更慢。"
        )
    )

    if inference_mode.startswith("快速诊断"):
        st.caption("推荐：保留完整融合诊断，仅关闭高开销 SHAP。")
    else:
        st.caption("用于需要查看局部特征贡献时。")

    st.markdown("### 📁 3. 数据源导入")
    uploaded_file = st.file_uploader(
        "上传基桩测线数据 (.csv / .txt)",
        type=["csv", "txt"]
    )

    st.markdown("### ⚙️ 4. 现场物理参量")
    velocity = st.number_input(
        "标段基准波速 (m/s)",
        value=3800,
        step=50
    )
    sample_interval = st.number_input(
        "检测采样间隔 (μs)",
        value=50,
        step=5
    )

    df = None
    pile_index = 0
    analyze_btn = False

    if uploaded_file is not None:
        try:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(
                    uploaded_file,
                    sep=r"\s+",
                    header=None
                )

            if len(df) == 0:
                raise ValueError("上传文件为空。")

            st.markdown("### 🔎 5. 选取具体桩号")
            pile_index = st.number_input(
                f"输入基桩编号 (0 ~ {len(df)-1})",
                min_value=0,
                max_value=len(df)-1,
                value=0,
                step=1
            )

            st.markdown("---")
            analyze_btn = st.button(
                "🚀 开始单桩独立分析",
                width='stretch',
                type="primary"
            )

        except Exception as e:
            st.error(f"文件读取错误：{e}")

    st.markdown("---")
    st.caption(
        "模型采用按需加载：只加载当前选择的分类模式和桩型。"
    )


# ============================================================
# 切换模式时清除上一组大模型缓存
# ============================================================
_current_model_key = (
    diagnosis_mode
    if diagnosis_mode == "多分类模式"
    else f"{diagnosis_mode}|{pile_type}"
)

if st.session_state.get("_active_model_key") != _current_model_key:
    try:
        load_model_bundle.clear()
    except Exception:
        pass
    st.session_state["_active_model_key"] = _current_model_key


# ============================================================
# 主界面
# ============================================================
st.title("🏗️ 桩基低应变完整性智能检测与诊断系统")

top_col1, top_col2 = st.columns([1.2, 2.8])

with top_col1:
    st.metric("当前分类任务", diagnosis_mode.replace("模式", ""))

with top_col2:
    if diagnosis_mode == "二分类模式":
        st.info(f"当前二分类专用模型：**{pile_type}**")
    else:
        st.info("当前使用：**5 类桩身完整性融合诊断模型**")


with st.expander("🛠️ 部署诊断信息（模型无法加载时查看）"):
    st.write("应用目录：", str(BASE_DIR))
    st.write("当前分类任务：", diagnosis_mode)
    if diagnosis_mode == "二分类模式":
        st.write("当前桩型：", pile_type)

    _diag_cfg = selected_config(diagnosis_mode, pile_type)
    _diag_rows = []
    for _kind in ["cnn", "scaler", "tabpfn"]:
        _found = first_existing(_diag_cfg[_kind])
        _diag_rows.append({
            "类型": _kind,
            "候选文件名": " | ".join(_diag_cfg[_kind]),
            "实际找到": _found.name if _found else "未找到"
        })

    st.dataframe(
        pd.DataFrame(_diag_rows),
        width='stretch'
    )


if uploaded_file is None:
    st.info("👈 请在左侧上传包含基桩数据的文件。")
    st.stop()

if df is None:
    st.stop()


# ============================================================
# 数据提取
# ============================================================
with st.expander(
    f"展开查看当前文件数据透视（共 {len(df)} 根桩）"
):
    st.dataframe(df.head(5), width='stretch')


wave_cols = [f"no{i}" for i in range(1, 257)]

if set(wave_cols).issubset(df.columns):
    wave_input = (
        df[wave_cols]
        .iloc[int(pile_index)]
        .values
        .astype(float)
    )
else:
    if df.shape[1] < 256:
        st.error(
            f"当前文件只有 {df.shape[1]} 列，无法提取 256 点波形。"
        )
        st.stop()

    wave_input = (
        df.iloc[int(pile_index), :256]
        .values
        .astype(float)
    )


# 现场特征优先使用文件中已有值
if set(PHYS_COLS).issubset(df.columns):
    phys_input = (
        df[PHYS_COLS]
        .iloc[int(pile_index)]
        .values
        .astype(float)
    )
else:
    # 注意：三个 scaler 的桩径均值约为 400 mm，
    # 因此默认桩径使用 400，而不是原程序中的 1.0。
    phys_input = np.array([
        400.0,        # Diameter, mm
        15.0,         # Length, m
        float(velocity),
        0.2,          # RMS
        5.0,          # Total_Energy
        1.5,          # Peak_to_Peak
        0.15,         # Abs_Mean
        3.0,          # Kurtosis
        0.1,          # Skewness
        0.2,          # Std_Dev
        1.2,          # Shape_Factor
        2.5,          # Crest_Factor
        3.0,          # Impulse_Factor
        3.5,          # Clearance_Factor
        0.1           # CV
    ], dtype=float)

    st.warning(
        "上传文件中未找到完整的 15 个物理/统计特征列，"
        "当前使用界面默认值。正式诊断建议上传训练时同格式的特征列。"
    )


# ============================================================
# 波形显示
# ============================================================
try:
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False
except Exception as _mpl_error:
    st.error(
        f"matplotlib 导入失败：{type(_mpl_error).__name__}: {_mpl_error}"
    )
    st.stop()

st.subheader(
    f"第 {int(pile_index)} 号基桩 - 时域低应变反射波曲线"
)

fig_wave, ax_wave = plt.subplots(
    figsize=(10, 2.6),
    dpi=120
)
ax_wave.plot(
    wave_input,
    color="#004488",
    linewidth=1.15
)
ax_wave.set_xlabel("采样点")
ax_wave.set_ylabel("响应")
ax_wave.grid(
    True,
    linestyle="--",
    alpha=0.35
)
plt.tight_layout()
st.pyplot(fig_wave, width='stretch')
plt.close(fig_wave)


# ============================================================
# 推理
# ============================================================
if analyze_btn:
    st.markdown("---")

    pile_key = (
        "__MULTI__"
        if diagnosis_mode == "多分类模式"
        else pile_type
    )

    with st.spinner(
        "正在加载所选模型并进行融合诊断..."
    ):
        try:
            bundle = load_model_bundle(
                diagnosis_mode,
                pile_key
            )
        except Exception as e:
            st.error("模型加载失败，但应用本身仍在运行。")
            st.exception(e)
            st.info(
                "请展开“部署诊断信息”，确认 CNN / Scaler / TabPFN "
                "在 GitHub 中的实际文件名。"
            )
            st.stop()

        cnn_model = bundle["cnn"]
        scaler = bundle["scaler"]
        tabpfn = bundle["tabpfn"]
        cfg = bundle["config"]

        s_val, k_val, f_ratio = extract_dynamic_features(
            wave_input
        )

        wave_cnn_input = np.asarray(
            wave_input,
            dtype=np.float32
        ).reshape(1, 256, 1)

        # ----------------------------
        # 多分类
        # ----------------------------
        if diagnosis_mode == "多分类模式":
            cnn_probs = np.asarray(
                cnn_model.predict(
                    wave_cnn_input,
                    verbose=0
                )[0],
                dtype=float
            ).reshape(-1)

            if cnn_probs.size != 5:
                st.error(
                    f"多分类 CNN 应输出5个概率，当前输出{cnn_probs.size}个。"
                )
                st.stop()

            super_features = build_multiclass_features(
                cnn_probs,
                phys_input,
                s_val,
                k_val,
                f_ratio
            )

        # ----------------------------
        # 二分类
        # ----------------------------
        else:
            cnn_prob_class1 = float(
                np.asarray(
                    cnn_model.predict(
                        wave_cnn_input,
                        verbose=0
                    )
                ).reshape(-1)[0]
            )

            super_features = build_binary_features(
                cnn_prob_class1,
                phys_input,
                f_ratio
            )

        # scaler
        x_scaled = scaler.transform(
            super_features
        )

        # TabPFN：只执行一次 predict_proba
        final_label, final_probs, classes, pred_pos = (
            get_final_prediction(
                tabpfn,
                x_scaled
            )
        )


    # ========================================================
    # 结果文字
    # ========================================================
    if diagnosis_mode == "多分类模式":
        diagnosis_map = MULTI_DIAGNOSIS_MAP
        result_text = diagnosis_map.get(
            final_label,
            f"类别 {final_label}"
        )

        labels_for_bar = [
            diagnosis_map.get(
                int(c),
                f"类别 {int(c)}"
            ).split(" ")[0]
            for c in classes
        ]

        chart_title = "5 大缺陷类别诊断概率分布"

    else:
        diagnosis_map = BINARY_DIAGNOSIS_MAP
        result_text = diagnosis_map.get(
            final_label,
            f"类别 {final_label}"
        )

        labels_for_bar = [
            diagnosis_map.get(
                int(c),
                f"类别 {int(c)}"
            )
            for c in classes
        ]

        chart_title = f"{pile_type}二分类诊断概率分布"

    confidence = float(
        final_probs[pred_pos]
    ) * 100.0


    # ========================================================
    # 结果状态框
    # ========================================================
    if final_label == 0:
        st.success(
            f"第 {int(pile_index)} 号基桩诊断结论："
            f"【{result_text}】，综合置信度：{confidence:.1f}%"
        )
    else:
        if diagnosis_mode == "多分类模式" and final_label in [1, 4]:
            st.warning(
                f"第 {int(pile_index)} 号基桩诊断结论："
                f"【{result_text}】，综合置信度：{confidence:.1f}%"
            )
        else:
            st.error(
                f"第 {int(pile_index)} 号基桩诊断结论："
                f"【{result_text}】，综合置信度：{confidence:.1f}%"
            )


    # ========================================================
    # 模型信息
    # ========================================================
    with st.expander("查看本次实际调用的模型文件"):
        st.write("CNN：", bundle["paths"]["cnn"])
        st.write("Scaler：", bundle["paths"]["scaler"])
        st.write("TabPFN：", bundle["paths"]["tabpfn"])
        st.write("CNN输出维度：", cfg["cnn_outputs"])
        st.write("融合特征维度：", cfg["n_features"])


    # ========================================================
    # 概率图 + 可选 SHAP
    # ========================================================
    if inference_mode.startswith("完整解释"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("诊断概率分布")
            render_probability_bar_chart(
                final_probs,
                labels_for_bar,
                pred_pos,
                chart_title
            )

        with col2:
            st.subheader("局部决策物理归因")

            with st.spinner("正在计算 SHAP 局部解释..."):
                try:
                    base_value, shap_values = (
                        calculate_shap_explanation(
                            tabpfn_model=tabpfn,
                            x_scaled=x_scaled,
                            n_features=cfg["n_features"],
                            predicted_position=pred_pos,
                            nsamples=20
                        )
                    )

                    render_waterfall_plot(
                        base_value=base_value,
                        shap_values=shap_values,
                        feature_names=cfg["feature_names"],
                        feature_values=x_scaled[0],
                        target_class_name=result_text
                    )

                except Exception as e:
                    st.warning(
                        "诊断已经完成，但 SHAP 解释生成失败："
                        f"{type(e).__name__}: {e}"
                    )

    else:
        st.subheader("诊断概率分布")
        render_probability_bar_chart(
            final_probs,
            labels_for_bar,
            pred_pos,
            chart_title
        )

        st.info(
            "当前为快速诊断模式：CNN + TabPFN 均参与最终判定，"
            "仅关闭 SHAP，因此不会因为关闭 SHAP 而改变诊断模型。"
        )
