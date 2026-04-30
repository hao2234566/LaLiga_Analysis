import os
import sys
import traceback

import streamlit as st
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_chinese_font():
    """
    配置 Matplotlib 中文字体，解决中文显示为方框的问题。
    优先使用 Streamlit Cloud/Linux 常见的 Noto CJK 字体。
    """
    font_candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS"
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}

    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break

    if selected_font:
        plt.rcParams["font.sans-serif"] = [selected_font]
        matplotlib.rcParams["font.sans-serif"] = [selected_font]
        print(f"已使用中文字体: {selected_font}")
    else:
        print("警告：未找到可用中文字体，中文可能显示为方框。")

    plt.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["axes.unicode_minus"] = False


setup_chinese_font()

# 保证可以导入 src 目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SRC_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

from src.data_loader import DataLoader
from src.team_analysis import TeamAnalyzer
from src.player_analysis import PlayerAnalyzer
from src.prediction_model import MatchPredictor


st.set_page_config(
    page_title="西甲数据分析与预测系统",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(show_spinner="正在加载并预处理数据...")
def load_all_data():
    """
    加载并预处理所有数据。
    使用 st.cache_data 避免每次点击页面都重新读 Excel。
    """
    loader = DataLoader(data_path=DATA_DIR)

    loader.load_current_season_team()
    loader.load_current_season_player()
    loader.load_historical_data()
    loader.load_round_data()

    team_data = loader.preprocess_team_data()
    player_data = loader.preprocess_player_data()

    historical_data = loader.historical_data
    round_data = loader.round_data

    return team_data, player_data, historical_data, round_data


@st.cache_resource(show_spinner="正在训练预测模型...")
def build_predictor(team_data, round_data):
    """
    训练比赛预测模型。
    用 cache_resource 缓存模型对象。
    """
    predictor = MatchPredictor(team_data, round_data)
    predictor.train_models()
    return predictor


def safe_run(func, *args, **kwargs):
    """
    安全执行绘图或分析函数，避免整个 Streamlit 页面崩掉。
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"执行失败：{e}")
        st.code(traceback.format_exc())
        return None


def show_saved_image(path, caption=None):
    """
    显示已经保存的图片。
    """
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"图片未生成：{path}")


def page_home(team_data, player_data):
    st.title("⚽ 西甲足球数据可视化分析与预测系统")

    st.markdown(
        """
        这是一个基于 **球队数据、球员数据、历史数据、逐轮比赛数据** 的西甲分析与预测系统。

        系统主要功能包括：

        - 联赛积分榜分析
        - 球队进攻、防守、传球、纪律、风格分析
        - 球员射手榜、助攻榜、xG、评分、效率分析
        - 基于 xG + 泊松分布的比赛结果预测
        - 比分概率热力图
        - 赛季冠军预测
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("球队数量", len(team_data))

    with col2:
        st.metric("球员数量", len(player_data))

    with col3:
        if "积分" in team_data.columns:
            leader = team_data.sort_values("积分", ascending=False).iloc[0]["球队"]
            st.metric("当前榜首", leader)
        else:
            st.metric("当前榜首", "未知")

    with col4:
        if "进球数" in player_data.columns and "球员姓名" in player_data.columns:
            top_scorer = player_data.sort_values("进球数", ascending=False).iloc[0]["球员姓名"]
            st.metric("射手榜第一", top_scorer)
        else:
            st.metric("射手榜第一", "未知")

    st.subheader("球队数据预览")
    st.dataframe(team_data, use_container_width=True)

    st.subheader("球员数据预览")
    st.dataframe(player_data.head(30), use_container_width=True)


def page_team_analysis(team_data, historical_data):
    st.title("📊 球队层面分析")

    analyzer = TeamAnalyzer(team_data, historical_data)

    chart_options = [
        "积分榜前10",
        "完整积分榜",
        "攻防雷达图",
        "进攻效率",
        "防守热力图",
        "传球分析",
        "进球失球分析",
        "球队风格",
        "历史对比",
        "纪律数据",
        "战术风格表"
    ]

    selected = st.selectbox("请选择分析图表", chart_options)

    if selected == "积分榜前10":
        path = os.path.join(OUTPUT_DIR, "st_积分榜前10.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_league_standings, save_path=path)
        show_saved_image(path, "西甲积分榜前10")

    elif selected == "完整积分榜":
        path = os.path.join(OUTPUT_DIR, "st_完整积分榜.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_all_teams_standings, save_path=path)
        show_saved_image(path, "完整积分榜")

    elif selected == "攻防雷达图":
        teams = st.multiselect(
            "选择最多4支球队",
            team_data["球队"].tolist(),
            default=team_data["球队"].head(4).tolist()
        )

        if len(teams) > 4:
            st.warning("雷达图最多建议选择4支球队。")
            teams = teams[:4]

        path = os.path.join(OUTPUT_DIR, "st_攻防雷达图.png")

        if st.button("生成雷达图"):
            safe_run(analyzer.plot_attack_defense_radar, teams=teams, save_path=path)

        show_saved_image(path, "球队攻防能力雷达图")

    elif selected == "进攻效率":
        path = os.path.join(OUTPUT_DIR, "st_进攻效率.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_offensive_comparison, save_path=path)
        show_saved_image(path, "球队进攻效率分析")

    elif selected == "防守热力图":
        path = os.path.join(OUTPUT_DIR, "st_防守热力图.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_defensive_heatmap, save_path=path)
        show_saved_image(path, "球队防守热力图")

    elif selected == "传球分析":
        path = os.path.join(OUTPUT_DIR, "st_传球分析.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_passing_analysis, save_path=path)
        show_saved_image(path, "球队传球能力分析")

    elif selected == "进球失球分析":
        path = os.path.join(OUTPUT_DIR, "st_进球失球分析.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_goals_analysis, save_path=path)
        show_saved_image(path, "进球与失球分析")

    elif selected == "球队风格":
        path = os.path.join(OUTPUT_DIR, "st_球队风格.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_team_style_scatter, save_path=path)
        show_saved_image(path, "球队风格分析")

    elif selected == "历史对比":
        path = os.path.join(OUTPUT_DIR, "st_历史对比.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_historical_comparison, save_path=path)
        show_saved_image(path, "历史赛季对比")

    elif selected == "纪律数据":
        path = os.path.join(OUTPUT_DIR, "st_纪律数据.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_discipline_analysis, save_path=path)
        show_saved_image(path, "纪律数据分析")

    elif selected == "战术风格表":
        style_df = safe_run(analyzer.team_style_analysis)
        if style_df is not None:
            st.dataframe(style_df, use_container_width=True)

            csv = style_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="下载战术风格分析 CSV",
                data=csv,
                file_name="球队战术风格分析.csv",
                mime="text/csv"
            )


def page_player_analysis(player_data):
    st.title("👤 球员层面分析")

    analyzer = PlayerAnalyzer(player_data)

    chart_options = [
        "射手榜",
        "助攻榜",
        "进球助攻分布",
        "xG分析",
        "评分分布",
        "评分最高球员",
        "射门效率",
        "出场时间",
        "位置对比",
        "球员雷达图",
        "最佳阵容"
    ]

    selected = st.selectbox("请选择球员分析内容", chart_options)

    if selected == "射手榜":
        path = os.path.join(OUTPUT_DIR, "st_射手榜.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_top_scorers, save_path=path)
        show_saved_image(path, "射手榜")

    elif selected == "助攻榜":
        path = os.path.join(OUTPUT_DIR, "st_助攻榜.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_top_assists, save_path=path)
        show_saved_image(path, "助攻榜")

    elif selected == "进球助攻分布":
        path = os.path.join(OUTPUT_DIR, "st_进球助攻分布.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_goal_assist_scatter, save_path=path)
        show_saved_image(path, "进球助攻分布")

    elif selected == "xG分析":
        path = os.path.join(OUTPUT_DIR, "st_xG分析.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_xg_analysis, save_path=path)
        show_saved_image(path, "xG分析")

    elif selected == "评分分布":
        path = os.path.join(OUTPUT_DIR, "st_评分分布.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_rating_distribution, save_path=path)
        show_saved_image(path, "评分分布")

    elif selected == "评分最高球员":
        path = os.path.join(OUTPUT_DIR, "st_评分最高球员.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_top_rated_players, save_path=path)
        show_saved_image(path, "评分最高球员")

    elif selected == "射门效率":
        path = os.path.join(OUTPUT_DIR, "st_射门效率.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_shooting_efficiency, save_path=path)
        show_saved_image(path, "射门效率")

    elif selected == "出场时间":
        path = os.path.join(OUTPUT_DIR, "st_出场时间.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_appearance_analysis, save_path=path)
        show_saved_image(path, "出场时间分析")

    elif selected == "位置对比":
        path = os.path.join(OUTPUT_DIR, "st_位置对比.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_position_comparison_boxplot, save_path=path)
        show_saved_image(path, "位置对比")

    elif selected == "球员雷达图":
        path = os.path.join(OUTPUT_DIR, "st_球员雷达图.png")
        if st.button("生成图表"):
            safe_run(analyzer.plot_player_radar, save_path=path)
        show_saved_image(path, "球员雷达图")

    elif selected == "最佳阵容":
        best_xi = safe_run(analyzer.get_best_xi)
        if best_xi is not None:
            st.subheader("本赛季最佳阵容")
            for pos, players in best_xi.items():
                st.markdown(f"### {pos}")
                st.dataframe(players, use_container_width=True)


def page_match_prediction(team_data, predictor):
    st.title("🔮 比赛结果预测")

    teams = team_data["球队"].tolist()

    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox("选择主队", teams, index=0)

    with col2:
        away_default_index = 1 if len(teams) > 1 else 0
        away_team = st.selectbox("选择客队", teams, index=away_default_index)

    if home_team == away_team:
        st.warning("主队和客队不能相同。")
        return

    if st.button("开始预测"):
        result = predictor.predict_match(home_team, away_team)

        if result is None:
            return

        st.subheader(f"{result['home_team']} vs {result['away_team']}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("主队预期进球 xG", result["home_xg"])

        with col2:
            st.metric("客队预期进球 xG", result["away_xg"])

        with col3:
            st.metric("最可能比分", result["most_likely_score"])

        st.subheader("胜平负概率")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.metric("主胜", result["probabilities"]["主胜"])

        with p2:
            st.metric("平局", result["probabilities"]["平局"])

        with p3:
            st.metric("客胜", result["probabilities"]["客胜"])

        st.info(f"模型预测结果：{result['prediction']}")

        st.subheader("双方基础数据")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"### {home_team}")
            st.json(result["home_stats"])

        with c2:
            st.markdown(f"### {away_team}")
            st.json(result["away_stats"])

        st.subheader("比分概率矩阵")
        score_df = result["score_probabilities"]
        st.dataframe(score_df, use_container_width=True)

        st.subheader("比分概率热力图")
        heatmap_path = os.path.join(
            OUTPUT_DIR,
            f"st_{home_team}_vs_{away_team}_比分热力图.png"
        )

        safe_run(
            predictor.plot_score_heatmap,
            home_team,
            away_team,
            save_path=heatmap_path
        )

        show_saved_image(heatmap_path, f"{home_team} vs {away_team} 比分热力图")


def page_model_analysis(predictor):
    st.title("🧠 模型分析")

    st.subheader("球队攻防强度")

    if st.button("生成球队攻防强度图"):
        path = os.path.join(OUTPUT_DIR, "st_球队攻防强度图.png")
        safe_run(predictor.plot_team_strengths, save_path=path)
        show_saved_image(path, "球队攻防强度图")

    st.subheader("模型内部强度数据")

    if predictor.strengths is not None:
        strength_df = pd.DataFrame.from_dict(predictor.strengths, orient="index")
        strength_df.index.name = "球队"
        strength_df = strength_df.reset_index()
        st.dataframe(strength_df, use_container_width=True)

        csv = strength_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="下载球队强度数据 CSV",
            data=csv,
            file_name="球队强度数据.csv",
            mime="text/csv"
        )
    else:
        st.warning("模型尚未训练。")


def page_champion_prediction(predictor):
    st.title("🏆 赛季冠军预测")

    remaining_matches = st.slider("剩余轮数", min_value=1, max_value=20, value=9)

    if st.button("预测冠军", type="primary"):
        standings = safe_run(predictor.predict_champion, remaining_matches=remaining_matches)

        if standings is not None:
            st.subheader("冠军预测结果")
            st.dataframe(standings, use_container_width=True)

            champion = standings.iloc[0]["球队"]
            expected_points = standings.iloc[0]["调整后预期积分"]

            st.success(f"预测冠军：{champion}，预期最终积分：{expected_points:.1f} 分")


def main():
    st.sidebar.title("导航")

    page = st.sidebar.radio(
        "请选择页面",
        [
            "首页",
            "球队分析",
            "球员分析",
            "比赛预测",
            "模型分析",
            "冠军预测"
        ]
    )

    try:
        team_data, player_data, historical_data, round_data = load_all_data()
    except Exception as e:
        st.error("数据加载失败，请检查 data 目录和 Excel 文件。")
        st.code(traceback.format_exc())
        return

    try:
        predictor = build_predictor(team_data, round_data)
    except Exception as e:
        st.error("预测模型训练失败。")
        st.code(traceback.format_exc())
        predictor = None

    if page == "首页":
        page_home(team_data, player_data)

    elif page == "球队分析":
        page_team_analysis(team_data, historical_data)

    elif page == "球员分析":
        page_player_analysis(player_data)

    elif page == "比赛预测":
        if predictor is None:
            st.error("预测模型不可用。")
        else:
            page_match_prediction(team_data, predictor)

    elif page == "模型分析":
        if predictor is None:
            st.error("预测模型不可用。")
        else:
            page_model_analysis(predictor)

    elif page == "冠军预测":
        if predictor is None:
            st.error("预测模型不可用。")
        else:
            page_champion_prediction(predictor)


if __name__ == "__main__":
    main()