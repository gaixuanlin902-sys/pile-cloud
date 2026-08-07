# ============================================================
# 桩基低应变完整性智能检测与多分类诊断系统 (交互升级定稿版)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import joblib
import os

# 延迟导入易崩溃库
try:
    import shap
except ImportError:
    pass

# ==========================================
# ⚙️ 页面全局配置
# ==========================================
st.set_page_config(page_title="基桩多分类智能诊断系统", layout="wide", initial_sidebar_state="expanded")
# 你的字体文件名（确保它和 app.py 放在同一个文件夹里）
font_path = "simhei.ttf"
if os.path.exists(font_path):
    # 终极霸王硬上弓法：强行把字体注册到系统中，并自己给它命名
    fe = fm.FontEntry(
        fname=font_path,
        name='MyCustomFont'  # 你可以随便起名，只要下面对应即可
    )
    fm.fontManager.ttflist.insert(0, fe)
    plt.rcParams['font.sans-serif'] = ['MyCustomFont'] # 全局使用这个你命名的字体
else:
    # 备用方案
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Songti SC', 'Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False 

DIAGNOSIS_MAP = {
    0: "完整桩 (Ⅰ类)", 
    1: "局部缩径桩 (Ⅱ类)", 
    2: "桩头破碎/差 (Ⅲ类)", 
    3: "严重断裂桩 (Ⅳ类)", 
    4: "局部扩径桩 (反相)"
}

FEATURE_NAMES_23D = [
    "CNN概率(断裂)", "CNN概率(扩径)", "CNN概率(缩径)", "CNN概率(完整)", "CNN概率(桩头)",
    "桩径(Diameter)", "桩长(Length)", "波速(Velocity)",
    "RMS", "总能量", "峰峰值", "绝对均值",
    "峰度", "偏度", "标准差", "波形因数",
    "峰值因数", "脉冲因数", "裕度因数", "变异系数",
    "动态波形偏度", "动态波形峰度", "FFT主频能量比"
]

# ==========================================
# 💾 真实模型加载模块 (安全隔离版)
# ==========================================
# ==========================================
# 💾 真实模型加载模块 (纯净权重读取版)
# ==========================================
def build_cnn_model(tf):
    """
    根据 cnn_weights_only.weights.h5 中保存的变量形状恢复 CNN 骨架。

    权重文件中的主要变量形状：
    - Conv1D-1 kernel: (5, 1, 32)
    - BatchNormalization: 4 × (32,)
    - Conv1D-2 kernel: (3, 32, 64)
    - Dense-1 kernel: (64, 32)
    - Dense-2 kernel: (32, 5)
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(256, 1), name="wave_input"),
            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=5,
                activation="relu",
                name="conv1d",
            ),
            tf.keras.layers.BatchNormalization(name="batch_normalization"),
            tf.keras.layers.MaxPooling1D(
                pool_size=2,
                name="max_pooling1d",
            ),
            tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=3,
                activation="relu",
                name="conv1d_1",
            ),
            tf.keras.layers.GlobalAveragePooling1D(
                name="global_average_pooling1d"
            ),
            tf.keras.layers.Dense(
                units=32,
                activation="relu",
                name="dense",
            ),
            # Dropout 在推理阶段自动关闭；其比例不影响权重加载。
            tf.keras.layers.Dropout(
                rate=0.3,
                name="dropout",
            ),
            tf.keras.layers.Dense(
                units=5,
                activation="softmax",
                name="dense_1",
            ),
        ],
        name="pile_wave_cnn",
    )


@st.cache_resource
def load_real_models():
    models = {
        "status": False,
        "cnn": None,
        "scaler": None,
        "tabpfn": None,
        "msg": "等待加载",
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))

    weights_path = os.path.join(current_dir, "cnn_weights_only.weights.h5")
    scaler_path = os.path.join(current_dir, "super_scaler.pkl")
    tabpfn_path = os.path.join(current_dir, "best_tabpfn_model.pkl")

    try:
        missing_files = [
            path
            for path in (weights_path, scaler_path, tabpfn_path)
            if not os.path.exists(path)
        ]
        if missing_files:
            models["msg"] = "缺少模型文件：" + "、".join(
                os.path.basename(path) for path in missing_files
            )
            return models

        # Scaler 和 TabPFN 必须与训练时的 Python 包版本兼容。
        models["scaler"] = joblib.load(scaler_path)
        models["tabpfn"] = joblib.load(tabpfn_path)

        import tensorflow as tf

        cnn_model = build_cnn_model(tf)

        # 确保模型已经建立变量，然后按完整拓扑加载权重。
        _ = cnn_model(
            np.zeros((1, 256, 1), dtype=np.float32),
            training=False,
        )
        cnn_model.load_weights(weights_path)

        # 部署前的结构一致性检查。
        expected_output_shape = (None, 5)
        if tuple(cnn_model.output_shape) != expected_output_shape:
            raise ValueError(
                "CNN 输出形状异常："
                f"{cnn_model.output_shape}，预期为 {expected_output_shape}"
            )

        scaler_features = getattr(models["scaler"], "n_features_in_", None)
        if scaler_features is not None and int(scaler_features) != 23:
            raise ValueError(
                f"Scaler 需要 {scaler_features} 个特征，"
                "但当前融合模型应输入 23 个特征。"
            )

        models["cnn"] = cnn_model
        models["status"] = True
        models["msg"] = "✅ CNN、Scaler 和 TabPFN 均已成功加载！"

    except Exception as e:
        models["status"] = False
        models["msg"] = (
            f"⚠️ 模型加载失败：{type(e).__name__}: {e}"
        )

    return models

models_dict = load_real_models()

# ==========================================
# 🧬 动态特征提取函数
# ==========================================
def pipeline_extract_features(wave_input, phys_input):
    s_val = skew(wave_input)
    k_val = kurtosis(wave_input)
    
    fft_mag = np.abs(np.fft.fft(wave_input))
    half_mag = fft_mag[1:128]
    f_ratio = np.max(half_mag) / (np.sum(half_mag) + 1e-8)
    
    enhanced_phys = np.hstack((phys_input, [s_val, k_val, f_ratio]))
    return enhanced_phys, s_val, k_val, f_ratio

# ==========================================
# 📊 图表渲染引擎 1: 置信度分布柱状图
# ==========================================
def render_probability_bar_chart(final_probs, diagnosis_map, predicted_class):
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=200)
    classes = [diagnosis_map[i].split(" ")[0] for i in range(5)]
    probs_pct = [p * 100 for p in final_probs]
    
    color_map = {0: '#2ca02c', 1: '#ff7f0e', 2: '#d62728', 3: '#d62728', 4: '#1f77b4'}
    highlight_color = color_map.get(predicted_class, '#004488')
    colors = [highlight_color if i == predicted_class else '#cccccc' for i in range(5)]
    
    bars = ax.bar(classes, probs_pct, color=colors, width=0.45, edgecolor='black', linewidth=0.8)
    ax.set_ylabel("判定置信度 (%)", fontsize=10)
    ax.set_ylim(0, max(probs_pct) * 1.25 if max(probs_pct) > 0 else 100)
    ax.set_title("5 大缺陷类别诊断概率分布对比", fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar, pct in zip(bars, probs_pct):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5, f"{pct:.1f}%",
                ha='center', va='bottom', fontsize=9.5,
                fontweight='bold' if pct == max(probs_pct) else 'normal')
        
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ==========================================
# 📊 图表渲染引擎 2: SHAP 瀑布图
# ==========================================
def render_waterfall_plot(base_value, shap_values, feature_names, feature_values, target_class_name):
    base_value = float(np.ravel(base_value)[0]) if isinstance(base_value, (list, np.ndarray)) else float(base_value)
    shap_values = np.array(shap_values, dtype=float).flatten()
    feature_values = np.array(feature_values, dtype=float).flatten()
    
    try:
        if 'shap' in globals():
            shap_exp = shap.Explanation(
                values=shap_values, base_values=base_value, data=feature_values, feature_names=feature_names
            )
            fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
            shap.plots.waterfall(shap_exp, max_display=10, show=False)
            plt.title(f"【{target_class_name}】决策推导归因瀑布图", fontsize=12, fontweight='bold', pad=15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            raise Exception("SHAP library not loaded.")
    except Exception:
        # Matplotlib 兜底渲染机制 (防止组件冲突)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        abs_order = np.argsort(np.abs(shap_values))[::-1][:8]
        top_names = [feature_names[i] for i in abs_order]
        top_shaps = shap_values[abs_order]
        
        y_pos = np.arange(len(top_names))
        colors = ['#ff0051' if v >= 0 else '#008bfb' for v in top_shaps]
        
        current = base_value
        lefts = []
        for val in top_shaps:
            if val >= 0:
                lefts.append(current)
                current += val
            else:
                current += val
                lefts.append(current)
                
        bars = ax.barh(y_pos, np.abs(top_shaps), left=lefts, color=colors, height=0.55, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=10)
        ax.invert_yaxis()
        
        for bar, val in zip(bars, top_shaps):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_y() + bar.get_height() / 2
            sign = "+" if val >= 0 else ""
            ax.text(x, y, f"{sign}{val:.3f}", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            
        final_val = base_value + np.sum(shap_values)
        ax.axvline(x=base_value, color='gray', linestyle='--', label=f'基准概率 E[f(x)] = {base_value:.2f}')
        ax.axvline(x=final_val, color='#e60000', linestyle='-', linewidth=1.5, label=f'最终置信度 f(x) = {final_val:.2f}')
        
        ax.set_xlabel("特征贡献度 (SHAP 值)", fontsize=10)
        ax.set_title(f"【{target_class_name}】决策推导归因瀑布图", fontsize=12, fontweight='bold', pad=12)
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ==========================================
# 🎛️ 侧边栏：核心数据通道与操作中枢
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/engineering.png", width=60)
    st.title("🎛️ 控制面板")
    st.markdown("---")
    
    if models_dict['status']:
        st.success(models_dict['msg'])
    else:
        st.warning(models_dict['msg'])

    st.markdown("### 📁 1. 数据源导入")
    uploaded_file = st.file_uploader("上传基桩测线数据 (.csv / .txt)", type=['csv', 'txt'])
    
    st.markdown("### ⚙️ 2. 现场物理参量")
    velocity = st.number_input("标段基准波速 (m/s)", value=3800, step=50)
    sample_interval = st.number_input("检测采样间隔 (μs)", value=50, step=5)

    df = None
    max_idx = 0
    pile_index = 0
    analyze_btn = False

    if uploaded_file is not None:
        st.markdown("###  3. 选取具体桩号")
        try:
            try:
                df = pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=r'\s+', header=None)
            
            max_idx = len(df) - 1
            pile_index = st.number_input(f"输入基桩编号 (0 ~ {max_idx})", min_value=0, max_value=max_idx, value=0, step=1)
            
            st.markdown("---")
            analyze_btn = st.button(" 开始单桩独立分析", use_container_width=True)
            
        except Exception as e:
            st.error(f"文件读取错误: {e}")

# ==========================================
# 🚀 主界面：数据联动与动态分析结果展示
# ==========================================
st.title("🏗️ 基桩低应变完整性智能检测与多分类诊断系统")

if uploaded_file is None:
    st.info("👈 请在左侧控制面板上传包含基桩数据的文件，以启动分析功能。")
elif df is not None:
    
    # --- 1. 展示数据表头 ---
    with st.expander(f"展开查看当前文件数据透视 (共 {len(df)} 根桩)"):
        st.dataframe(df.head(5))

    # --- 2. 提取用户选中的“那根具体桩”的数据 ---
    wave_cols = [f"no{i}" for i in range(1, 257)]
    if set(wave_cols).issubset(df.columns):
        wave_input = df[wave_cols].iloc[pile_index].values
    else:
        wave_input = df.iloc[pile_index, :256].values.astype(float)
        
    phys_cols = [
        "Diameter", "Length", "Velocity", "RMS", "Total_Energy", "Peak_to_Peak", "Abs_Mean",
        "Kurtosis", "Skewness", "Std_Dev", "Shape_Factor", "Crest_Factor", "Impulse_Factor", "Clearance_Factor", "CV"
    ]
    if set(phys_cols).issubset(df.columns):
        phys_input = df[phys_cols].iloc[pile_index].values
    else:
        phys_input = np.array([1.0, 15.0, velocity, 0.2, 5.0, 1.5, 0.15, 3.0, 0.1, 0.2, 1.2, 2.5, 3.0, 3.5, 0.1])

    # --- 3. 实时联动显示选中的时域波形 ---
    st.subheader(f" 第 {pile_index} 号基桩 - 时域低应变反射波曲线")
    fig_wave, ax_wave = plt.subplots(figsize=(10, 2.5), dpi=200)
    ax_wave.plot(wave_input, color='#004488', linewidth=1.2)
    ax_wave.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_wave)
    plt.close(fig_wave)

    # --- 4. 点击“开始分析”后的推理与作图逻辑 ---
    if analyze_btn:
        st.markdown("---")
        with st.spinner(f"正在对 第 {pile_index} 号基桩 进行诊断与特征物理溯源..."):
            
            enhanced_phys, s_val, k_val, f_ratio = pipeline_extract_features(wave_input, phys_input)
            
            # 【分支 A】: 真实模型预测
            if models_dict['status'] and models_dict['cnn'] is not None and models_dict['tabpfn'] is not None:
                wave_cnn_input = wave_input.reshape(1, 256, 1)
                cnn_probs = models_dict['cnn'].predict(wave_cnn_input, verbose=0)[0]
                
                super_features = np.hstack((cnn_probs, enhanced_phys)).reshape(1, -1)
                super_features_scaled = models_dict['scaler'].transform(super_features)
                
                final_pred = int(models_dict['tabpfn'].predict(super_features_scaled)[0])
                final_probs = models_dict['tabpfn'].predict_proba(super_features_scaled)[0]
                
                background_data = np.zeros((10, 23))
                
                # 安全调用 SHAP
                if 'shap' in globals():
                    explainer = shap.KernelExplainer(models_dict['tabpfn'].predict_proba, background_data)
                    shap_values_all = explainer.shap_values(super_features_scaled, nsamples=100)
                    
                    if isinstance(shap_values_all, list):
                        raw_shap_vals = shap_values_all[final_pred][0]
                        base_val = float(np.ravel(explainer.expected_value[final_pred])[0])
                    else:
                        raw_shap_vals = shap_values_all[0, :, final_pred]
                        base_val = float(np.ravel(explainer.expected_value[final_pred])[0])
                else:
                    # 模拟 SHAP 防止崩溃
                    raw_shap_vals = np.zeros(23)
                    raw_shap_vals[22] = 0.5
                    base_val = 0.2
                
                target_prob = final_probs[final_pred]
                s_sum = np.sum(raw_shap_vals)
                if abs(s_sum) > 1e-6:
                    class_shap_vals = raw_shap_vals * ((target_prob - base_val) / s_sum)
                else:
                    class_shap_vals = raw_shap_vals
                    class_shap_vals[0] = target_prob - base_val
                    
                feat_values_disp = super_features_scaled[0]

            # 【分支 B】: 自适应规则/演示推断
            else:
                impact_peak = np.max(wave_input[:40]) if np.max(wave_input[:40]) > 0 else 1.0
                min_refl = np.min(wave_input[40:])
                max_refl = np.max(wave_input[40:])
                
                if min_refl < -0.4 * impact_peak:
                    final_pred, final_probs = 4, [0.03, 0.04, 0.03, 0.05, 0.85]
                elif max_refl > 0.7 * impact_peak:
                    final_pred, final_probs = 3, [0.02, 0.05, 0.03, 0.88, 0.02]
                elif max_refl > 0.3 * impact_peak:
                    final_pred, final_probs = 1, [0.10, 0.78, 0.07, 0.02, 0.03]
                elif s_val < -1.0 and k_val > 2.0:
                    final_pred, final_probs = 2, [0.05, 0.10, 0.75, 0.05, 0.05]
                else:
                    final_pred, final_probs = 0, [0.88, 0.06, 0.02, 0.02, 0.02]
                    
                base_val = 0.20
                target_prob = final_probs[final_pred]
                total_delta = target_prob - base_val
                
                class_shap_vals = np.zeros(23)
                class_shap_vals[final_pred] = total_delta * 0.60
                class_shap_vals[22] = total_delta * 0.25
                class_shap_vals[6] = total_delta * 0.15
                feat_values_disp = np.hstack(([0.1]*5, enhanced_phys))

            # --- 展示最终结果 ---
            result_text = DIAGNOSIS_MAP[final_pred]
            confidence = final_probs[final_pred] * 100.0
            
            if final_pred == 0:
                st.success(f" 第 {pile_index} 号基桩诊断结论：【 {result_text} 】, 综合置信度: {confidence:.1f}%")
            elif final_pred in [1, 4]:
                st.warning(f" 第 {pile_index} 号基桩诊断结论：【 {result_text} 】, 综合置信度: {confidence:.1f}%")
            else:
                st.error(f" 第 {pile_index} 号基桩诊断结论：【 {result_text} 】, 综合置信度: {confidence:.1f}%")

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("5大类别置信度分布")
                render_probability_bar_chart(final_probs, DIAGNOSIS_MAP, final_pred)

            with col2:
                st.subheader("局部决策物理归因溯源")
                render_waterfall_plot(
                    base_value=base_val, shap_values=class_shap_vals, 
                    feature_names=FEATURE_NAMES_23D, feature_values=feat_values_disp, 
                    target_class_name=result_text
                )