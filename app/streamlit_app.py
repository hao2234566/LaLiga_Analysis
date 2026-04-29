import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# =========================
# 路径配置（适配app子文件夹存放，彻底解决导入问题）
# =========================
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# 精准定位项目根目录（LaLiga_Analysis文件夹）
CURRENT_FILE_PATH = Path(__file__).resolve()  # 当前文件：app/streamlit_app.py
PROJECT_ROOT = CURRENT_FILE_PATH.parents[1]  # 往上跳1层到app，再跳1层到项目根目录

# 把项目根目录和src目录加入Python环境，确保能正常导入模块
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

# 导入模块（路径修复后才能正常导入）
from src.data_loader import DataLoader
from src.prediction_model import MatchPredictor
# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="西甲球队分析与比赛预测系统",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 页面样式
# =========================
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #16324f;
        margin-bottom: 8px;
    }
    .sub-title {
        font-size: 18px;
        color: #5b6b7a;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 28px;
        font-weight: 700;
        color: #1b3a57;
        margin-top: 8px;
        margin-bottom: 15px;
    }
    .card {
        background: linear-gradient(135deg, #f8fbff, #eef5fc);
        padding: 18px 20px;
        border-radius: 14px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
        border-left: 6px solid #1f77b4;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 15px;
        color: #5c677d;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1f2d3d;
    }
    .info-box {
        background-color: #eef6ff;
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #d6e9ff;
        color: #1f2d3d;
        margin-bottom: 15px;
    }
    .small-note {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 数据加载
# =========================
@st.cache_data
def load_all_data():
    loader = DataLoader(data_path=str(PROJECT_ROOT / "data"))
    # 加载基础数据
    loader.load_current_season_team()
    team_data = loader.preprocess_team_data()
    loader.load_current_season_player()
    player_data = loader.preprocess_player_data()
    round_data = loader.load_round_data()
    historical_data = loader.load_historical_data()
    # 新增：初始化并训练预测模型，和本地逻辑完全对齐
    predictor = MatchPredictor(team_data, round_data)
    predictor.train_models()
    return team_data, player_data, round_data, historical_data, predictor

# =========================
# 工具函数
# =========================
def show_metric_card(col, title, value):
    with col:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


def safe_bar_chart(df, x_col, y_col, title, color=None, height=520):
    if x_col in df.columns and y_col in df.columns:
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            color=color if color and color in df.columns else None,
            title=title,
            text_auto=True
        )
        fig.update_layout(
            title_x=0.5,
            height=height,
            xaxis_title="",
            yaxis_title=""
        )
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)


def safe_scatter(df, x_col, y_col, title, hover_name=None, color=None, height=620):
    if x_col in df.columns and y_col in df.columns:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            hover_name=hover_name if hover_name in df.columns else None,
            color=color if color and color in df.columns else None,
            title=title,
            height=height
        )
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)



# =========================
# 页面：首页
# =========================
def page_home(team_data, player_data):
    st.markdown('<div class="main-title">⚽ 西甲球队分析与比赛预测系统</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">基于西甲球队数据、球员数据、历史赛季数据的综合可视化分析平台</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
    本系统集成球队表现分析、球员个人数据分析、历史赛季对比分析、逐轮比赛数据分析与比赛结果预测功能，
    可用于毕业设计展示、联赛趋势研究及足球数据可视化应用。
    </div>
    """, unsafe_allow_html=True)

    total_teams = len(team_data)
    total_players = len(player_data)
    total_matches = int(team_data['场次'].sum() / 2) if '场次' in team_data.columns else 0
    total_goals = int(team_data['进球'].sum()) if '进球' in team_data.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    show_metric_card(c1, "参赛球队数量", total_teams)
    show_metric_card(c2, "统计球员数量", total_players)
    show_metric_card(c3, "已统计比赛场次", total_matches)
    show_metric_card(c4, "联赛总进球数", total_goals)

    st.markdown('<div class="section-title">联赛整体概览</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        safe_bar_chart(
            team_data.sort_values(by='积分', ascending=False),
            '球队',
            '积分',
            '球队积分榜',
            color='积分'
        )
    with col2:
        if '进球' in team_data.columns and '场次' in team_data.columns:
            team_data_copy = team_data.copy()
            team_data_copy['场均进球'] = team_data_copy['进球'] / team_data_copy['场次']
            safe_bar_chart(
                team_data_copy.sort_values(by='场均进球', ascending=False),
                '球队',
                '场均进球',
                '球队场均进球对比',
                color='场均进球'
            )

    st.subheader("球队基础数据表")
    st.dataframe(team_data, use_container_width=True, height=420)


# =========================
# 页面：球队分析
# =========================
def page_team_analysis(team_data):
    st.markdown('<div class="section-title">球队数据分析</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    本模块用于从进攻、防守、组织等多个角度分析球队表现，
    并通过交互式图表展示各球队之间的差异。
    </div>
    """, unsafe_allow_html=True)

    if '场次' in team_data.columns:
        team_data = team_data.copy()
        if '进球' in team_data.columns:
            team_data['场均进球'] = team_data['进球'] / team_data['场次']
        if '失球' in team_data.columns:
            team_data['场均失球'] = team_data['失球'] / team_data['场次']
        if '射门' in team_data.columns:
            team_data['场均射门'] = team_data['射门'] / team_data['场次']
        if '射正' in team_data.columns:
            team_data['场均射正'] = team_data['射正'] / team_data['场次']
        if '抢断' in team_data.columns:
            team_data['场均抢断'] = team_data['抢断'] / team_data['场次']
        if '拦截' in team_data.columns:
            team_data['场均拦截'] = team_data['拦截'] / team_data['场次']

    numeric_cols = team_data.select_dtypes(include='number').columns.tolist()
    x_axis = st.selectbox("选择 X 轴指标", numeric_cols, index=0)
    y_axis = st.selectbox("选择 Y 轴指标", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

    safe_scatter(
        team_data,
        x_axis,
        y_axis,
        f'{x_axis} 与 {y_axis} 的关系分析',
        hover_name='球队',
        color='积分' if '积分' in team_data.columns else None
    )

    st.markdown('<div class="small-note">说明：散点图采用悬停显示球队名称，避免标签重叠。</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if '球队' in team_data.columns and '场均进球' in team_data.columns:
            safe_bar_chart(
                team_data.sort_values(by='场均进球', ascending=False),
                '球队',
                '场均进球',
                '球队场均进球排名',
                color='场均进球'
            )
    with col2:
        if '球队' in team_data.columns and '场均失球' in team_data.columns:
            safe_bar_chart(
                team_data.sort_values(by='场均失球', ascending=True),
                '球队',
                '场均失球',
                '球队场均失球排名',
                color='场均失球'
            )

    st.subheader("单支球队详细画像")
    selected_team = st.selectbox("选择球队", team_data['球队'].tolist())
    selected_df = team_data[team_data['球队'] == selected_team]
    st.dataframe(selected_df, use_container_width=True)

    display_cols = [col for col in ['场均进球', '场均失球', '场均射门', '场均射正', '场均抢断', '场均拦截', '射正率', '传球成功率'] if col in selected_df.columns]
    if display_cols:
        detail_df = selected_df[display_cols].T.reset_index()
        detail_df.columns = ['指标', '数值']
        safe_bar_chart(detail_df, '指标', '数值', f'{selected_team} 关键指标分析', color='数值', height=500)


# =========================
# 页面：球员分析
# =========================
def page_player_analysis(player_data):
    st.markdown('<div class="section-title">球员数据分析</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    本模块展示球员个人表现数据，包括进球、助攻、期望进球、射门、评分、防守数据等，
    可用于球员能力评估与赛季表现分析。
    </div>
    """, unsafe_allow_html=True)

    df = player_data.copy()

    # 计算衍生指标
    if '出场次数' in df.columns:
        if '进球数' in df.columns:
            df['场均进球'] = df['进球数'] / df['出场次数']
        if '助攻数' in df.columns:
            df['场均助攻'] = df['助攻数'] / df['出场次数']
        if '总射门次数' in df.columns:
            df['场均射门'] = df['总射门次数'] / df['出场次数']
        if '射正次数' in df.columns:
            df['场均射正'] = df['射正次数'] / df['出场次数']

    if '总射门次数' in df.columns and '进球数' in df.columns:
        df['进球转化率'] = df.apply(lambda x: x['进球数'] / x['总射门次数'] if x['总射门次数'] != 0 else 0, axis=1)

    col1, col2, col3 = st.columns(3)

    teams = ['全部']
    if '联赛' in df.columns:
        leagues = ['全部'] + sorted(df['联赛'].dropna().unique().tolist())
    else:
        leagues = ['全部']

    if '球员位置' in df.columns:
        positions = ['全部'] + sorted(df['球员位置'].dropna().unique().tolist())
    else:
        positions = ['全部']

    with col1:
        selected_league = st.selectbox("选择联赛", leagues)
    with col2:
        selected_position = st.selectbox("选择球员位置", positions)
    with col3:
        min_apps = st.slider("最少出场次数", 0, 38, 5)

    if selected_league != '全部' and '联赛' in df.columns:
        df = df[df['联赛'] == selected_league]

    if selected_position != '全部' and '球员位置' in df.columns:
        df = df[df['球员位置'] == selected_position]

    if '出场次数' in df.columns:
        df = df[df['出场次数'] >= min_apps]

    st.subheader("球员数据表")
    st.dataframe(df, use_container_width=True, height=420)

    # 排行榜
    rank_options = [col for col in ['进球数', '助攻数', '期进球(xG)', '期助攻(xA)', '球员评分', '场均进球', '场均助攻', '场均射门', '进球转化率'] if col in df.columns]

    if rank_options:
        rank_col = st.selectbox("选择排行榜指标", rank_options)
        top_n = st.slider("显示前 N 名", 5, 20, 10)

        top_df = df.sort_values(by=rank_col, ascending=False).head(top_n)
        safe_bar_chart(
            top_df,
            '球员姓名',
            rank_col,
            f'球员{rank_col} Top {top_n}',
            color=rank_col,
            height=550
        )

    # 球员效率分布
    if '场均射门' in df.columns and '场均进球' in df.columns:
        safe_scatter(
            df,
            '场均射门',
            '场均进球',
            '球员射门与进球效率分布',
            hover_name='球员姓名',
            color='球员位置' if '球员位置' in df.columns else None
        )

    # 进球助攻对比
    if '进球数' in df.columns and '助攻数' in df.columns:
        compare_df = df[['球员姓名', '进球数', '助攻数']].sort_values(by='进球数', ascending=False).head(10)
        compare_df = compare_df.melt(id_vars='球员姓名', var_name='指标', value_name='数值')

        fig = px.bar(
            compare_df,
            x='球员姓名',
            y='数值',
            color='指标',
            barmode='group',
            title='球员进球与助攻对比 Top 10',
            height=550
        )
        fig.update_layout(title_x=0.5)
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    # 单球员详细画像
    st.subheader("单球员详细画像")
    selected_player = st.selectbox("选择球员", df['球员姓名'].tolist())
    selected_player_df = df[df['球员姓名'] == selected_player]
    st.dataframe(selected_player_df, use_container_width=True)

    player_cols = [col for col in ['进球数', '助攻数', '期进球(xG)', '期助攻(xA)', '球员评分', '总射门次数', '射正次数', '抢断次数', '拦截次数', '扑救次数'] if col in selected_player_df.columns]
    if player_cols:
        detail_df = selected_player_df[player_cols].T.reset_index()
        detail_df.columns = ['指标', '数值']
        safe_bar_chart(detail_df, '指标', '数值', f'{selected_player} 关键指标分析', color='数值', height=500)


# =========================
# 页面：历史赛季
# =========================
def page_history(historical_data):
    st.markdown('<div class="section-title">历史赛季数据分析</div>', unsafe_allow_html=True)

    if not historical_data:
        st.warning("没有加载到历史赛季数据。")
        return

    seasons = sorted(historical_data.keys())
    selected_season = st.selectbox("选择赛季", seasons)

    hist_df = historical_data[selected_season]
    st.dataframe(hist_df, use_container_width=True, height=420)

    if '阵容' in hist_df.columns and '积分' in hist_df.columns:
        safe_bar_chart(
            hist_df.sort_values(by='积分', ascending=False),
            '阵容',
            '积分',
            f'{selected_season} 赛季积分榜',
            color='积分'
        )


# =========================
# 页面：逐轮比赛
# =========================
def page_rounds(round_data):
    st.markdown('<div class="section-title">逐轮比赛数据分析</div>', unsafe_allow_html=True)

    if not round_data:
        st.warning("没有加载到逐轮比赛数据。")
        return

    round_keys = sorted(round_data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    selected_round = st.selectbox("选择轮次", round_keys)

    round_df = round_data[selected_round]
    st.dataframe(round_df, use_container_width=True, height=420)

    col1, col2 = st.columns(2)

    with col1:
        if '主队' in round_df.columns and '进球数' in round_df.columns:
            safe_bar_chart(round_df, '主队', '进球数', f'第 {selected_round} 轮主队进球情况', color='进球数', height=500)

    with col2:
        if '客队' in round_df.columns and '客队进球数' in round_df.columns:
            safe_bar_chart(round_df, '客队', '客队进球数', f'第 {selected_round} 轮客队进球情况', color='客队进球数', height=500)


# =========================
# 页面：比赛预测
# =========================
def page_prediction(team_data, predictor):
    st.markdown('<div class="section-title">比赛结果预测</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    本模块基于xG预期进球模型+泊松分布混合模型，结合球队攻防强度、近期状态、主客场优势等多维度特征，
    对比赛结果、预期进球、比分概率进行专业预测，和本地端模型完全一致。
    </div>
    """, unsafe_allow_html=True)

    teams = team_data['球队'].tolist()
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("选择主队", teams)
    with col2:
        away_team = st.selectbox("选择客队", teams, index=1 if len(teams) > 1 else 0)

    if home_team == away_team:
        st.warning("主队和客队不能相同。")
        return

    if st.button("开始预测", use_container_width=True):
        # 调用和本地完全一致的预测方法
        result = predictor.predict_match(home_team, away_team, detailed=True)

        # 核心结果卡片
        c1, c2, c3 = st.columns(3)
        show_metric_card(c1, "预测结果", result['prediction'])
        show_metric_card(c2, "最可能比分", result['most_likely_score'])
        show_metric_card(c3, "预期进球(xG)", f"{result['home_xg']} : {result['away_xg']}")

        # 概率分布柱状图
        st.subheader("结果概率分布")
        prob_df = pd.DataFrame({
            '结果': ['主胜', '平局', '客胜'],
            '概率': [
                float(result['probabilities']['主胜'].replace('%', '')),
                float(result['probabilities']['平局'].replace('%', '')),
                float(result['probabilities']['客胜'].replace('%', ''))
            ]
        })
        fig = px.bar(prob_df, x='结果', y='概率', color='结果', text_auto='.1f', title='比赛结果概率分布')
        fig.update_layout(title_x=0.5, height=400)
        st.plotly_chart(fig, use_container_width=True)

        # 比分概率热力图
        st.subheader("详细比分概率热力图")
        score_matrix = result['score_probabilities']
        fig_heatmap = px.imshow(
            score_matrix,
            labels=dict(x=f"{away_team}进球数", y=f"{home_team}进球数", color="概率(%)"),
            x=score_matrix.columns,
            y=score_matrix.index,
            color_continuous_scale='YlOrRd',
            text_auto='.1f'
        )
        fig_heatmap.update_layout(title=f"{home_team} vs {away_team} 比分概率分布", title_x=0.5, height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # 双方指标对比
        st.subheader("双方核心指标对比")
        compare_cols = ['积分', '场均进球', '场均失球', '估算xG']
        home_row = team_data[team_data['球队'] == home_team][compare_cols].iloc[0]
        away_row = team_data[team_data['球队'] == away_team][compare_cols].iloc[0]
        detail_df = pd.DataFrame({
            '指标': compare_cols,
            home_team: home_row.values,
            away_team: away_row.values
        }).melt(id_vars='指标', var_name='球队', value_name='数值')
        fig_compare = px.bar(
            detail_df,
            x='指标',
            y='数值',
            color='球队',
            barmode='group',
            title='双方关键指标对比',
            height=520
        )
        fig_compare.update_layout(title_x=0.5)
        st.plotly_chart(fig_compare, use_container_width=True)

# =========================
# 主函数
# =========================
def main():
    try:
        team_data, player_data, round_data, historical_data, predictor = load_all_data()
    except Exception as e:
        st.error(f"数据加载失败：{e}")
        st.stop()

    st.sidebar.markdown("## 功能导航")
    page = st.sidebar.radio(
        "请选择页面",
        [
            "系统首页",
            "球队数据分析",
            "球员数据分析",
            "历史赛季数据",
            "逐轮比赛数据",
            "比赛结果预测"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
**毕业设计项目**  
题目：基于机器学习的西甲球队分析与比赛预测系统  

**系统模块：**
- 系统首页
- 球队数据分析
- 球员数据分析
- 历史赛季数据
- 逐轮比赛数据
- 比赛结果预测
""")

    if page == "系统首页":
        page_home(team_data, player_data)
    elif page == "球队数据分析":
        page_team_analysis(team_data)
    elif page == "球员数据分析":
        page_player_analysis(player_data)
    elif page == "历史赛季数据":
        page_history(historical_data)
    elif page == "逐轮比赛数据":
        page_rounds(round_data)
    elif page == "比赛结果预测":
        page_prediction(team_data, predictor)


if __name__ == "__main__":
    main()
    