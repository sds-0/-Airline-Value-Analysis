import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")

    print("\n数据基本信息:")
    df.info()

    missing_values = df.isnull().sum()
    print("\n缺失值统计:")
    print(missing_values[missing_values > 0])

    for col in ['SUM_YR_1', 'SUM_YR_2']:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
            print(f"已用均值填充 {col} 的缺失值")

    df = df[df['SUM_YR_1'].notnull() & df['SUM_YR_2'].notnull()]
    index1 = (df['SUM_YR_1'] != 0) | (df['SUM_YR_2'] != 0)
    index2 = (df['SEG_KM_SUM'] == 0) & (df['avg_discount'] == 0)
    index3 = df['AGE'] <= 100 if 'AGE' in df.columns else True
    df = df[(index1 | index2) & index3]
    df = df.dropna(subset=['FFP_DATE', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount'])

    print(f"预处理后数据形状: {df.shape}")
    return df

def explore_data_distribution(df):
    df['FFP_YEAR'] = pd.to_datetime(df['FFP_DATE']).dt.year
    plt.figure(figsize=(10, 5))
    sns.histplot(df['FFP_YEAR'], bins=10, kde=True, color='steelblue')
    plt.title('会员入会年份分布')
    plt.xlabel('年份')
    plt.ylabel('会员人数')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df['FLIGHT_COUNT'], color='lightgreen')
    plt.title('客户飞行次数分布')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    corr_cols = ['FLIGHT_COUNT', 'SEG_KM_SUM', 'LAST_TO_END', 'avg_discount']
    corr = df[corr_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('关键特征相关性热力图')
    plt.tight_layout()
    plt.show()


def calculate_lrfmc(df):
    df['FFP_DATE'] = pd.to_datetime(df['FFP_DATE'])
    df['LOAD_TIME'] = pd.to_datetime(df['LOAD_TIME'])

    df['L'] = ((df['LOAD_TIME'] - df['FFP_DATE']).dt.days / 30).round(2)  # 会员时长（月）
    df['R'] = df['LAST_TO_END']  # 最近乘机间隔（天）
    df['F'] = df['FLIGHT_COUNT']  # 飞行频率
    df['M'] = df['SEG_KM_SUM']  # 总飞行里程
    df['C'] = df['avg_discount']  # 平均折扣率

    lrfmc = df[['L', 'R', 'F', 'M', 'C']].copy()
    print("\nLRFMC特征统计描述:")
    print(lrfmc.describe().round(2))
    return lrfmc


def standardize_features(lrfmc):
    scaler = StandardScaler()
    lrfmc_std = scaler.fit_transform(lrfmc)
    return pd.DataFrame(lrfmc_std, columns=lrfmc.columns)


def perform_kmeans_clustering(lrfmc_std_df, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(lrfmc_std_df)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    lrfmc_std_df['cluster'] = labels

    cluster_counts = pd.Series(labels).value_counts().sort_index().reset_index()
    cluster_counts.columns = ['cluster', 'count']
    total = len(lrfmc_std_df)
    cluster_counts['percentage'] = (cluster_counts['count'] / total * 100).round(2)

    centers_df = pd.DataFrame(centers, columns=lrfmc_std_df.columns[:-1])
    centers_df['cluster'] = range(k)
    cluster_analysis = pd.merge(centers_df, cluster_counts, on='cluster')

    return lrfmc_std_df, centers, cluster_analysis


def analyze_cluster_features(lrfmc, lrfmc_std_df, k):
    lrfmc_std_df_with_cluster, centers, cluster_analysis = perform_kmeans_clustering(lrfmc_std_df, k)

    print("\n聚类分析结果:")
    print(cluster_analysis[['cluster', 'count', 'percentage', 'L', 'R', 'F', 'M', 'C']])

    plot_radar_chart(cluster_analysis, k)

    plot_feature_heatmap(cluster_analysis, k)

    return cluster_analysis


def plot_radar_chart(cluster_analysis, k):
    plt.rcParams['figure.dpi'] = 300
    features = ['L', 'R', 'F', 'M', 'C']
    data = cluster_analysis[features].values

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    data = np.concatenate((data, data[:, [0]]), axis=1)
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i in range(k):
        ax.plot(angles, data[i], 'o-', linewidth=2, color=colors[i], label=f'聚类{i}')
        ax.fill(angles, data[i], alpha=0.1, color=colors[i])

    ax.set_thetagrids(np.degrees(angles[:-1]), features)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('各聚类客户LRFMC特征雷达图')
    plt.tight_layout()
    plt.show()


def plot_feature_heatmap(cluster_analysis, k):
    plt.figure(figsize=(10, 6))
    features = ['L', 'R', 'F', 'M', 'C']
    heatmap_data = cluster_analysis[features].copy()
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f',
                xticklabels=features, yticklabels=[f'聚类{i}' for i in range(k)])
    plt.title('各聚类客户特征热力图')
    plt.tight_layout()
    plt.show()


def evaluate_customer_value(cluster_analysis):
    cluster_analysis['customer_type'] = '一般与低价值客户'

    mask_keep = (cluster_analysis['F'] > 1) & (cluster_analysis['M'] > 1) & (cluster_analysis['R'] < -0.5)
    cluster_analysis.loc[mask_keep, 'customer_type'] = '重要保持客户'

    mask_develop = (cluster_analysis['C'] > 1) & (cluster_analysis['R'] < -0.3) & (cluster_analysis['F'] < 0)
    cluster_analysis.loc[mask_develop, 'customer_type'] = '重要发展客户'

    mask_retain = (cluster_analysis['F'] > 0.5) & (cluster_analysis['M'] > 0.5) & (cluster_analysis['R'] > 1)
    cluster_analysis.loc[mask_retain, 'customer_type'] = '重要挽留客户'

    print("\n客户价值分析结果:")
    print(cluster_analysis[['cluster', 'count', 'percentage', 'customer_type']])

    plot_value_distribution(cluster_analysis)
    return cluster_analysis[['cluster', 'customer_type']]


import plotly.express as px


def plot_value_distribution(cluster_analysis):
    data = {
        '客户类型': ['高价值客户', '高价值客户', '高价值客户', '低价值客户'],
        '细分类型': ['重要保持客户', '重要发展客户', '重要挽留客户', '一般与低价值客户'],
        '数量': [
            cluster_analysis[cluster_analysis['customer_type'] == '重要保持客户']['count'].sum(),
            cluster_analysis[cluster_analysis['customer_type'] == '重要发展客户']['count'].sum(),
            cluster_analysis[cluster_analysis['customer_type'] == '重要挽留客户']['count'].sum(),
            cluster_analysis[cluster_analysis['customer_type'] == '一般与低价值客户']['count'].sum()
        ]
    }

    fig = px.sunburst(
        data,
        path=['客户类型', '细分类型'],
        values='数量',
        color='细分类型',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        title='客户价值层级分布'
    )

    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.show()


def generate_marketing_strategies(cluster_analysis, value_ranking):
    strategies = []
    for _, row in cluster_analysis.iterrows():
        cluster = int(row['cluster'])
        cust_type = value_ranking[value_ranking['cluster'] == cluster]['customer_type'].values[0]

        if cust_type == '重要保持客户':
            strategy = (f"针对{cust_type}（聚类{cluster}）的策略：\n"
                        f"  - 专属权益：免费升舱、机场VIP通道（匹配高C特征）\n"
                        f"  - 保级提醒：距离保级差{int(50 - row['F'])}次飞行，推荐近期热门航线\n"
                        f"  - 忠诚度奖励：积分加倍累积（强化高F/M行为）")

        elif cust_type == '重要发展客户':
            strategy = (f"针对{cust_type}（聚类{cluster}）的策略：\n"
                        f"  - 频率激励：多程套餐买3送1（提升飞行次数F）\n"
                        f"  - 偏好匹配：推送高折扣舱位航线（符合高C偏好）\n"
                        f"  - 会员升级路径：明确告知升级后专属服务（引导长期留存）")

        elif cust_type == '重要挽留客户':
            strategy = (f"针对{cust_type}（聚类{cluster}）的策略：\n"
                        f"  - 召回激励：赠送{int(row['M'] * 0.1)}里程（激活沉默客户）\n"
                        f"  - 原因调研：电话回访了解未乘机原因（针对性解决）\n"
                        f"  - 回归礼包：首次复飞享额外积分（降低复飞门槛）")

        else:
            strategy = (f"针对{cust_type}（聚类{cluster}）的策略：\n"
                        f"  - 低成本触达：仅推送特价航线和淡季折扣（控制营销成本）\n"
                        f"  - 首次兑换引导：降低积分兑换门槛（促进首次互动）\n"
                        f"  - 筛选潜力客户：对高C低F用户尝试定向优惠券（精准转化）")

        strategies.append(strategy)
    return strategies


def main():
    file_path = r"C:\Users\Administrator\Desktop\homework\dasan2\data mining\shixun\air_data[dataset 2].csv"

    print("===== 步骤1：数据加载与预处理 =====")
    df = load_and_preprocess_data(file_path)

    print("\n===== 步骤2：数据探索分析 =====")
    explore_data_distribution(df)

    print("\n===== 步骤3：计算LRFMC特征 =====")
    lrfmc = calculate_lrfmc(df)

    print("\n===== 步骤4：特征标准化 =====")
    lrfmc_std_df = standardize_features(lrfmc)

    print("\n===== 步骤5：K-means聚类分析 =====")
    k = 5
    cluster_analysis = analyze_cluster_features(lrfmc, lrfmc_std_df, k)

    print("\n===== 步骤6：客户价值评估 =====")
    customer_value = evaluate_customer_value(cluster_analysis)

    print("\n===== 步骤7：生成营销策略 =====")
    strategies = generate_marketing_strategies(cluster_analysis, customer_value)

    print("\n===== 最终营销策略推荐 =====")
    for strategy in strategies:
        print("\n" + strategy)


if __name__ == "__main__":
    main()