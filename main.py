"""
西甲数据分析主程序（修复闪退+强制输出版）
新增：模型评估与特征重要性分析模块
"""
import os
import sys
# ========== 修复1：强制设置matplotlib后端，彻底解决弹窗闪退 ==========
import matplotlib
matplotlib.use('Agg')  # 必须放在所有import plt之前，禁用弹窗，只保存图片

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.data_loader import DataLoader
from src.team_analysis import TeamAnalyzer
from src.player_analysis import PlayerAnalyzer
from src.prediction_model import MatchPredictor
from src.model_evaluation import run_model_evaluation

def main():
    # ========== 修复2：所有print加flush=True，强制立即输出，不被缓冲隐藏 ==========
    print("=" * 60, flush=True)
    print("西甲足球数据可视化分析与预测系统", flush=True)
    print("=" * 60, flush=True)

    # 创建输出目录
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ 输出目录已创建/确认: {output_dir}", flush=True)

    # 1. 数据加载（新增逐轮数据加载+完整报错打印）
    print("\n【1】正在加载数据...", flush=True)
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    loader = DataLoader(data_path=data_path)
    try:
        loader.load_current_season_team()
        print("  ✅ 球队数据加载成功", flush=True)
        loader.load_current_season_player()
        print("  ✅ 球员数据加载成功", flush=True)
        loader.load_historical_data()
        print("  ✅ 历史数据加载成功", flush=True)
        loader.load_round_data()
        print("  ✅ 逐轮数据加载成功", flush=True)
    except Exception as e:
        print(f"\n❌ 数据加载失败，完整报错信息：", flush=True)
        import traceback
        traceback.print_exc()  # 打印完整错误堆栈，精准定位问题
        return

    # 2. 数据预处理
    print("\n【2】正在预处理数据...", flush=True)
    team_data = loader.preprocess_team_data()
    player_data = loader.preprocess_player_data()
    print("  ✅ 数据预处理完成", flush=True)

    # 3. 球队分析
    print("\n【3】正在进行球队层面分析...", flush=True)
    team_analyzer = TeamAnalyzer(team_data, loader.historical_data)
    try:
        team_analyzer.plot_league_standings(save_path=f'{output_dir}/01_积分榜.png')
        team_analyzer.plot_all_teams_standings(save_path=f'{output_dir}/02_积分榜全部.png')
        team_analyzer.plot_attack_defense_radar(save_path=f'{output_dir}/03_攻防雷达图.png')
        team_analyzer.plot_offensive_comparison(save_path=f'{output_dir}/04_进攻效率.png')
        team_analyzer.plot_defensive_heatmap(save_path=f'{output_dir}/05_防守热力图.png')
        team_analyzer.plot_passing_analysis(save_path=f'{output_dir}/06_传球分析.png')
        team_analyzer.plot_goals_analysis(save_path=f'{output_dir}/07_进球失球分析.png')
        team_analyzer.plot_team_style_scatter(save_path=f'{output_dir}/08_球队风格.png')
        team_analyzer.plot_historical_comparison(save_path=f'{output_dir}/09_历史对比.png')
        team_analyzer.plot_discipline_analysis(save_path=f'{output_dir}/10_纪律数据.png')
        # 保存战术风格分析结果
        style_df = team_analyzer.team_style_analysis()
        style_df.to_excel(f'{output_dir}/球队战术风格分析.xlsx', index=False)
        print("  ✅ 球队分析全部完成，图表已保存", flush=True)
    except Exception as e:
        print(f"\n❌ 球队分析失败，完整报错信息：", flush=True)
        import traceback
        traceback.print_exc()
        return

    # 4. 球员分析
    print("\n【4】正在进行球员层面分析...", flush=True)
    player_analyzer = PlayerAnalyzer(player_data)
    try:
        player_analyzer.plot_top_scorers(save_path=f'{output_dir}/11_射手榜.png')
        player_analyzer.plot_top_assists(save_path=f'{output_dir}/12_助攻榜.png')
        player_analyzer.plot_goal_assist_scatter(save_path=f'{output_dir}/13_进球助攻分布.png')
        player_analyzer.plot_xg_analysis(save_path=f'{output_dir}/14_xG分析.png')
        player_analyzer.plot_rating_distribution(save_path=f'{output_dir}/15_评分分布.png')
        player_analyzer.plot_top_rated_players(save_path=f'{output_dir}/16_评分最高.png')
        player_analyzer.plot_shooting_efficiency(save_path=f'{output_dir}/17_射门效率.png')
        player_analyzer.plot_appearance_analysis(save_path=f'{output_dir}/18_出场时间.png')
        player_analyzer.plot_position_comparison_boxplot(save_path=f'{output_dir}/19_位置对比.png')
        player_analyzer.plot_player_radar(save_path=f'{output_dir}/20_球员雷达图.png')
        # 输出最佳阵容
        best_xi = player_analyzer.get_best_xi()
        print("\n  ✅ 球员分析全部完成，本赛季最佳阵容:", flush=True)
        for pos, players in best_xi.items():
            print(f"    {pos}: {', '.join(players['球员姓名'].tolist())}", flush=True)
    except Exception as e:
        print(f"\n❌ 球员分析失败，完整报错信息：", flush=True)
        import traceback
        traceback.print_exc()
        return

    # 5. 预测模型（传入逐轮数据，启用近期状态修正）
    print("\n【5】正在构建预测模型...", flush=True)
    try:
        predictor = MatchPredictor(team_data, loader.round_data)
        predictor.train_models()
        predictor.plot_feature_importance(save_path=f'{output_dir}/21_特征重要性.png')
        print("  ✅ 模型训练完成", flush=True)

        # ========== 新增：自动生成你需要的两张图表 ==========
        # 1. 所有球队攻防强度气泡图
        print("  - 正在生成球队攻防强度气泡图...", flush=True)
        predictor.plot_team_strengths(save_path=f'{output_dir}/22_球队攻防强度图.png')
        # 2. 皇马vs巴萨比分热力图
        print("  - 正在生成皇马vs巴萨比分热力图...", flush=True)
        home_team = "皇家马德里"
        away_team = "巴塞罗那"
        predictor.plot_score_heatmap(home_team, away_team, save_path=f'{output_dir}/23_皇马vs巴萨_比分热力图.png')
        print("  ✅ 预测模型图表生成完成", flush=True)
    except Exception as e:
        print(f"\n❌ 预测模型构建失败，完整报错信息：", flush=True)
        import traceback
        traceback.print_exc()
        return

    # ========== 新增：模型评估与特征重要性分析 ==========
    print("\n【6】正在进行模型评估与特征重要性分析...", flush=True)
    try:
        evaluator, eval_results, cv_results = run_model_evaluation(
            team_data,
            loader.round_data,
            output_dir=output_dir
        )
        print("  ✅ 模型评估完成", flush=True)
    except Exception as e:
        print(f"\n❌ 模型评估失败，完整报错信息：", flush=True)
        import traceback
        traceback.print_exc()
        # 评估失败不中断主流程

    # 经典对决预测
    print("\n【7】经典对决预测结果", flush=True)
    matches = [('巴塞罗那', '皇家马德里'), ('马德里竞技', '巴塞罗那'), ('皇家马德里', '马德里竞技')]
    for home, away in matches:
        try:
            result = predictor.predict_match(home_team, away_team)
            print(f"\n📊 {result['home_team']} vs {result['away_team']}", flush=True)
            print(f"   当前积分: {result['home_stats']['积分']} - {result['away_stats']['积分']}", flush=True)
            print(f"   预期进球(xG): {result['home_xg']} - {result['away_xg']}", flush=True)
            print(f"   预测结果: {result['prediction']}", flush=True)
            print(f"   概率: 主胜{result['probabilities']['主胜']} | 平局{result['probabilities']['平局']} | 客胜{result['probabilities']['客胜']}", flush=True)
        except Exception as e:
            print(f"    {home} vs {away}: 预测失败 - {e}", flush=True)

    # 赛季冠军预测
    print("\n【8】赛季冠军预测", flush=True)
    predictor.predict_champion(remaining_matches=9)

    # 交互式预测（可选，不需要可以注释掉）
    print("\n【9】进入交互式预测界面", flush=True)
    predictor.interactive_prediction()

    # 最终完成提示
    print("\n" + "=" * 60, flush=True)
    print(f"🎉 全部分析完成！共生成26张图表，全部保存至 {output_dir}/ 目录", flush=True)
    print("=" * 60, flush=True)

if __name__ == '__main__':
    main()
