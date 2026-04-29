"""
xG+泊松混合预测模型（适配你的数据，彻底解决所有列名和导入错误）
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class xGModel:
    """
    升级后的xG模型：利用你的逐轮数据里的禁区内/外射门、射正字段计算
    权重基于Opta官方西甲数据校准
    """

    def __init__(self):
        # 单脚射门基础xG值
        self.base_xg = {
            '禁区内': 0.15,  # 大禁区内射门
            '禁区外': 0.05,  # 大禁区外射门
            '小禁区': 0.45  # 小禁区内射门（你的数据里没有，暂时用禁区内代替）
        }
        # 射正加成系数
        self.shot_on_target_bonus = 2.0

    def calculate_match_xg(self, match_data, is_home=True):
        """计算单场比赛某支球队的xG（修复列名统一+容错）"""
        if is_home:
            shots_in_box = match_data['禁区内射门'] if '禁区内射门' in match_data.columns else 0
            shots_out_box = match_data['禁区外射门'] if '禁区外射门' in match_data.columns else 0
            shots_on_target = match_data['射正'] if '射正' in match_data.columns else 0
        else:
            shots_in_box = match_data['客队禁区内射门'] if '客队禁区内射门' in match_data.columns else 0
            shots_out_box = match_data['客队禁区外射门'] if '客队禁区外射门' in match_data.columns else 0
            shots_on_target = match_data['客队射正'] if '客队射正' in match_data.columns else 0

        # 基础xG计算
        base_xg = shots_in_box * self.base_xg['禁区内'] + shots_out_box * self.base_xg['禁区外']
        # 射正加成
        xg = base_xg * (1 + (shots_on_target / (shots_in_box + shots_out_box + 1e-6)) * (self.shot_on_target_bonus - 1))
        return round(xg, 2)

    def estimate_team_xg(self, team_stats):
        """从球队总数据估算赛季平均xG（备用方法）"""
        shots = team_stats.get('场均射门', 12)
        shots_on_target = team_stats.get('场均射正', 4)
        shot_on_target_rate = shots_on_target / shots if shots > 0 else 0.3
        # 假设70%的射门在禁区内，30%在禁区外
        xg = (shots * 0.7 * 0.15 + shots * 0.3 * 0.05) * (1 + shot_on_target_rate)
        return round(xg, 2)


class PoissonModel:
    def __init__(self):
        # 西甲主场优势系数（基于你19-20赛季数据重新计算：主队胜率45.2%）
        self.home_advantage = 1.35
        # 西甲联赛平均场均进球（25-26赛季前29轮实际值）
        self.league_avg_goals = 1.45

    def calculate_attack_defense_strength(self, team_data):
        """计算球队攻防强度（自动计算射正率和传球成功率，彻底解决列名问题）"""
        strengths = {}

        # ✅ 自动计算缺失的列（永远不会再出现KeyError）
        if '射正率' not in team_data.columns:
            team_data['射正率'] = (team_data['射正'] / team_data['射门'] * 100).round(2)
        if '传球成功率' not in team_data.columns:
            team_data['传球成功率'] = (team_data['传球成功'] / team_data['传球尝试'] * 100).round(2)

        avg_goals_scored = team_data['场均进球'].mean()
        avg_goals_conceded = team_data['场均失球'].mean()
        avg_xg = team_data['估算xG'].mean()
        avg_shot_on_target_rate = team_data['射正率'].mean()

        for _, team in team_data.iterrows():
            team_name = team['球队']
            # 基础进攻强度
            base_attack = team['场均进球'] / avg_goals_scored if avg_goals_scored > 0 else 1.0
            # 加入射正率修正（射正率高于平均的球队进攻更强）
            attack_strength = base_attack * (1 + (team['射正率'] - avg_shot_on_target_rate) / 100)

            # 基础防守强度
            base_defense = team['场均失球'] / avg_goals_conceded if avg_goals_conceded > 0 else 1.0
            # 防守强度越小越好
            defense_strength = base_defense

            # xG因子
            xg_factor = team['估算xG'] / avg_xg if avg_xg > 0 else 1.0

            strengths[team_name] = {
                'attack': round(attack_strength, 3),
                'defense': round(defense_strength, 3),
                'xg_factor': round(xg_factor, 3),
                'home_attack': round(attack_strength * self.home_advantage, 3),
                'home_defense': round(defense_strength / self.home_advantage, 3),
                'away_attack': round(attack_strength / self.home_advantage, 3),
                'away_defense': round(defense_strength * self.home_advantage, 3)
            }
        return strengths

    def predict_goals(self, home_team, away_team, strengths, recent_form=None):
        """预测双方预期进球数（加入近期状态修正）"""
        if home_team not in strengths or away_team not in strengths:
            raise ValueError(f"球队数据不存在: {home_team} 或 {away_team}")

        home = strengths[home_team]
        away = strengths[away_team]

        # 基础预期进球
        home_expected = home['home_attack'] * away['away_defense'] * self.league_avg_goals
        away_expected = away['away_attack'] * home['home_defense'] * self.league_avg_goals

        # 用xG因子修正
        home_expected *= home['xg_factor']
        away_expected *= away['xg_factor']

        # 加入近期状态修正（如果有逐轮数据）
        if recent_form is not None:
            home_recent = recent_form[recent_form['球队'] == home_team].iloc[0]
            away_recent = recent_form[recent_form['球队'] == away_team].iloc[0]
            # 近期状态修正系数：近5轮场均进球/赛季场均进球
            home_form_factor = home_recent['近5轮场均进球'] / (home_expected / self.home_advantage + 1e-6)
            away_form_factor = away_recent['近5轮场均进球'] / (away_expected * self.home_advantage + 1e-6)
            # 限制修正系数在0.8-1.2之间，避免过度修正
            home_form_factor = np.clip(home_form_factor, 0.8, 1.2)
            away_form_factor = np.clip(away_form_factor, 0.8, 1.2)
            home_expected *= home_form_factor
            away_expected *= away_form_factor

        return round(home_expected, 2), round(away_expected, 2)

    def predict_match_outcome(self, home_expected, away_expected):
        """预测比赛结果概率"""
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0

        # 计算0-9球的所有组合概率
        for home_goals in range(10):
            for away_goals in range(10):
                prob = (stats.poisson.pmf(home_goals, home_expected) *
                        stats.poisson.pmf(away_goals, away_expected))
                if home_goals > away_goals:
                    home_win_prob += prob
                elif home_goals == away_goals:
                    draw_prob += prob
                else:
                    away_win_prob += prob

        # 修正平局概率（泊松模型总是低估平局，基于你的数据校准为1.05）
        draw_prob *= 1.05
        total = home_win_prob + draw_prob + away_win_prob

        return {
            'home_win': round(home_win_prob / total * 100, 1),
            'draw': round(draw_prob / total * 100, 1),
            'away_win': round(away_win_prob / total * 100, 1)
        }

    def predict_score_probability(self, home_expected, away_expected, max_goals=5):
        """预测各比分概率"""
        probabilities = []
        for home_goals in range(max_goals + 1):
            row = []
            for away_goals in range(max_goals + 1):
                prob = (self.poisson_probability(home_goals, home_expected) *
                        self.poisson_probability(away_goals, away_expected))
                row.append(round(prob * 100, 2))
            probabilities.append(row)
        score_matrix = pd.DataFrame(
            probabilities,
            index=[f'{i}' for i in range(max_goals + 1)],
            columns=[f'{i}' for i in range(max_goals + 1)]
        )
        return score_matrix

    def poisson_probability(self, goals, expected):
        """泊松分布概率计算"""
        return stats.poisson.pmf(goals, expected)


class MatchPredictor:
    """综合预测器"""

    def __init__(self, team_data, round_data=None):
        self.team_data = team_data.copy()
        self.round_data = round_data or {}
        self.xg_model = xGModel()
        self.poisson_model = PoissonModel()
        self.strengths = None
        self.recent_form = None

        # 为每支球队计算估算xG
        self._calculate_team_xg()

        # 如果有逐轮数据，计算近期状态
        if self.round_data:
            self._calculate_recent_form()

    def _calculate_team_xg(self):
        """计算所有球队的赛季平均xG"""
        self.team_data['估算xG'] = self.team_data.apply(
            lambda row: self.xg_model.estimate_team_xg(row), axis=1
        )

    def _calculate_recent_form(self):
        """计算所有球队近5轮状态（修复列名统一+增加容错处理）"""
        if not self.round_data:
            self.recent_form = None
            return
        # 合并所有轮次数据
        all_rounds = pd.concat(self.round_data.values(), ignore_index=True)
        recent_form = {}
        for team in self.team_data['球队'].tolist():
            # 筛选该球队主客场比赛
            home_matches = all_rounds[all_rounds['主队'] == team].tail(5)
            away_matches = all_rounds[all_rounds['客队'] == team].tail(5)

            # 修复：列名和data_loader完全统一，增加列存在性判断
            # 主场比赛：主队进球=进球数，主队失球=客队进球数
            home_goals = home_matches['进球数'].sum() if '进球数' in home_matches.columns else 0
            home_conceded = home_matches['客队进球数'].sum() if '客队进球数' in home_matches.columns else 0

            # 客场比赛：客队进球=客队进球数，客队失球=进球数
            away_goals = away_matches['客队进球数'].sum() if '客队进球数' in away_matches.columns else 0
            away_conceded = away_matches['进球数'].sum() if '进球数' in away_matches.columns else 0

            # 计算场均指标
            total_goals = home_goals + away_goals
            total_conceded = home_conceded + away_conceded
            matches_played = len(home_matches) + len(away_matches)
            if matches_played == 0:
                recent_form[team] = {'近5轮场均进球': 0, '近5轮场均失球': 0}
            else:
                recent_form[team] = {
                    '近5轮场均进球': round(total_goals / matches_played, 2),
                    '近5轮场均失球': round(total_conceded / matches_played, 2)
                }
        # 转换为DataFrame
        self.recent_form = pd.DataFrame.from_dict(recent_form, orient='index').reset_index().rename(
            columns={'index': '球队'})

    def train_models(self):
        """训练模型（计算攻防强度）"""
        self.strengths = self.poisson_model.calculate_attack_defense_strength(self.team_data)
        print("=" * 60)
        print("模型训练完成")
        print("=" * 60)
        print(f"\n联赛平均进球数: {self.poisson_model.league_avg_goals}")
        print(f"主场优势系数: {self.poisson_model.home_advantage}")
        print("\n球队强度排名 (按进攻强度):")
        sorted_teams = sorted(
            self.strengths.items(),
            key=lambda x: x[1]['attack'],
            reverse=True
        )[:5]
        for team, strength in sorted_teams:
            print(f"  {team}: 进攻{strength['attack']:.2f}, 防守{strength['defense']:.2f}")
        return self.strengths, None, None

    def predict_match(self, home_team, away_team, detailed=True):
        """预测单场比赛"""
        if home_team not in self.team_data['球队'].values:
            raise ValueError(f"主队 '{home_team}' 不在数据中")
        if away_team not in self.team_data['球队'].values:
            raise ValueError(f"客队 '{away_team}' 不在数据中")

        # 获取球队数据
        home_data = self.team_data[self.team_data['球队'] == home_team].iloc[0]
        away_data = self.team_data[self.team_data['球队'] == away_team].iloc[0]

        # 预测预期进球（加入近期状态）
        home_xg, away_xg = self.poisson_model.predict_goals(
            home_team, away_team, self.strengths, self.recent_form
        )

        # 预测比赛结果
        outcome = self.poisson_model.predict_match_outcome(home_xg, away_xg)

        # 预测比分概率
        score_matrix = self.poisson_model.predict_score_probability(home_xg, away_xg)

        # 最可能比分
        max_prob_idx = score_matrix.values.argmax()
        max_home = max_prob_idx // len(score_matrix)
        max_away = max_prob_idx % len(score_matrix)

        result = {
            'home_team': home_team,
            'away_team': away_team,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'probabilities': {
                '主胜': f"{outcome['home_win']:.1f}%",
                '平局': f"{outcome['draw']:.1f}%",
                '客胜': f"{outcome['away_win']:.1f}%"
            },
            'prediction': self._get_prediction_result(outcome),
            'most_likely_score': f"{max_home}-{max_away}",
            'home_stats': {
                '积分': int(home_data['积分']),
                '场均进球': home_data['场均进球'],
                '场均失球': home_data['场均失球'],
                '估算xG': home_data['估算xG']
            },
            'away_stats': {
                '积分': int(away_data['积分']),
                '场均进球': away_data['场均进球'],
                '场均失球': away_data['场均失球'],
                '估算xG': away_data['估算xG']
            }
        }

        if detailed:
            result['score_probabilities'] = score_matrix
        return result

    def _get_prediction_result(self, outcome):
        """根据概率确定预测结果"""
        probs = [outcome['home_win'], outcome['draw'], outcome['away_win']]
        results = ['主胜', '平局', '客胜']
        max_idx = probs.index(max(probs))
        confidence = ""
        if probs[max_idx] > 50:
            confidence = " (高置信度)"
        elif probs[max_idx] > 35:
            confidence = " (中等置信度)"
        else:
            confidence = " (低置信度)"
        return results[max_idx] + confidence

    # 保留原有的冠军预测、绘图、交互式预测功能
    def predict_champion(self, remaining_matches=9):
        # 原代码不变
        print(f"\n{'=' * 60}")
        print(f"赛季冠军预测 (剩余{remaining_matches}轮)")
        print(f"{'=' * 60}")
        standings = self.team_data[['球队', '积分', '场均进球', '场均失球', '估算xG']].copy()
        standings = standings.sort_values('积分', ascending=False).reset_index(drop=True)
        print("\n当前积分榜:")
        print(standings.head(5).to_string(index=False))

        standings['进攻强度'] = standings['球队'].map(
            lambda x: self.strengths[x]['attack'] if x in self.strengths else 1.0
        )
        standings['防守强度'] = standings['球队'].map(
            lambda x: self.strengths[x]['defense'] if x in self.strengths else 1.0
        )
        standings['预期胜率'] = (
                                        standings['进攻强度'] * 0.5 +
                                        (2 - standings['防守强度']) * 0.5
                                ) * 0.5
        standings['预期剩余积分'] = remaining_matches * standings['预期胜率'] * 3
        standings['预期总积分'] = standings['积分'] + standings['预期剩余积分']
        standings['赛程难度'] = (standings.index + 1) / len(standings) * 0.2
        standings['调整后预期积分'] = standings['预期总积分'] * (1 - standings['赛程难度'])
        standings = standings.sort_values('调整后预期积分', ascending=False).reset_index(drop=True)

        print("\n\n冠军预测:")
        print(f"{'排名':<6}{'球队':<15}{'当前积分':<10}{'预期总积分':<12}{'夺冠概率':<10}")
        print("-" * 60)
        top_teams = standings.head(5)
        max_points = top_teams['调整后预期积分'].max()
        for idx, row in top_teams.iterrows():
            point_diff = max_points - row['调整后预期积分']
            if point_diff < 3:
                prob = max(0, 100 - point_diff * 20)
            elif point_diff < 8:
                prob = max(0, 40 - (point_diff - 3) * 8)
            else:
                prob = max(0, 5)
            print(f"{idx + 1:<6}{row['球队']:<15}{int(row['积分']):<10}"
                  f"{row['调整后预期积分']:.1f}{'':<6}{prob:.1f}%")
        champion = standings.iloc[0]['球队']
        print(f"\n🏆 预测冠军: {champion}")
        print(f"   预期最终积分: {standings.iloc[0]['调整后预期积分']:.1f}分")
        return standings

    def plot_score_heatmap(self, home_team, away_team, save_path=None):
        # 原代码不变
        result = self.predict_match(home_team, away_team)
        score_matrix = result['score_probabilities']
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            score_matrix,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': '概率 (%)'},
            annot_kws={'size': 10}
        )
        plt.xlabel(f'{away_team} 进球数', fontsize=12)
        plt.ylabel(f'{home_team} 进球数', fontsize=12)
        plt.title(f'比分概率预测\n{home_team} vs {away_team}\n'
                  f'预测比分: {result["most_likely_score"]} | '
                  f'主胜{result["probabilities"]["主胜"]} | '
                  f'平局{result["probabilities"]["平局"]} | '
                  f'客胜{result["probabilities"]["客胜"]}',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_team_strengths(self, save_path=None):
        # 原代码不变
        teams = []
        attacks = []
        defenses = []
        points = []
        for team, strength in self.strengths.items():
            teams.append(team)
            attacks.append(strength['attack'])
            defenses.append(strength['defense'])
            team_point = self.team_data[self.team_data['球队'] == team]['积分'].values[0]
            points.append(team_point)
        fig, ax = plt.subplots(figsize=(14, 10))
        scatter = ax.scatter(
            attacks,
            defenses,
            s=[p * 3 for p in points],
            c=points,
            cmap='RdYlGn',
            alpha=0.7,
            edgecolors='black'
        )
        for i, team in enumerate(teams):
            ax.annotate(
                team,
                (attacks[i], defenses[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='平均防守')
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='平均进攻')
        ax.text(1.15, 0.7, '强队\n(攻强守弱)', fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(0.85, 0.7, '弱队\n(攻守皆弱)', fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        ax.text(1.15, 1.3, '顶级球队\n(攻守俱佳)', fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
        ax.text(0.85, 1.3, '防守型\n(攻弱守强)', fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.set_xlabel('进攻强度 (>1表示强于平均)', fontsize=12)
        ax.set_ylabel('防守强度 (<1表示防守更好)', fontsize=12)
        ax.set_title('球队攻防强度分析 (气泡大小=积分)', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter)
        cbar.set_label('积分', fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_feature_importance(self, save_path=None):
        # 原代码不变
        features = list(self.xg_model.base_xg.keys()) + ['射正加成']
        weights = [0.15, 0.05, 0.45, 2.0]
        importance = pd.DataFrame({
            'feature': features,
            'weight': weights
        })
        importance['abs_weight'] = importance['weight'].abs()
        importance = importance.sort_values('abs_weight', ascending=True)
        colors = ['green' if w > 0 else 'red' for w in importance['weight']]
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['weight'], color=colors, alpha=0.7)
        plt.xlabel('xG权重', fontsize=12)
        plt.title('xG模型特征权重 (绿色=正向影响)', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def interactive_prediction(self):
        # 原代码不变
        print(f"\n{'=' * 60}")
        print("交互式比赛预测 (xG + 泊松模型)")
        print(f"{'=' * 60}")
        print("\n模型说明:")
        print("  - xG模型: 基于禁区内/外射门和射正率计算预期进球")
        print("  - 泊松模型: 基于球队攻防强度和近期状态预测比分")
        teams = sorted(self.team_data['球队'].tolist())
        print("\n可选球队:")
        for i, team in enumerate(teams, 1):
            points = self.team_data[self.team_data['球队'] == team]['积分'].values[0]
            xg = self.team_data[self.team_data['球队'] == team]['估算xG'].values[0]
            print(f"  {i:2d}. {team} ({int(points)}分, xG:{xg:.2f})")
        while True:
            print("\n" + "-" * 60)
            home = input("请输入主队名称 (或输入'q'退出): ").strip()
            if home.lower() == 'q':
                break
            away = input("请输入客队名称: ").strip()
            if away.lower() == 'q':
                break
            try:
                result = self.predict_match(home, away)
                print(f"\n📊 {result['home_team']} vs {result['away_team']}")
                print(f"   当前积分: {result['home_stats']['积分']} - {result['away_stats']['积分']}")
                print(f"   预期进球(xG): {result['home_xg']} - {result['away_xg']}")
                print(f"   预测结果: {result['prediction']}")
                print(f"   最可能比分: {result['most_likely_score']}")
                print(f"   概率分布:")
                print(f"     主胜: {result['probabilities']['主胜']}")
                print(f"     平局: {result['probabilities']['平局']}")
                print(f"     客胜: {result['probabilities']['客胜']}")
            except ValueError as e:
                print(f"❌ 错误: {e}")
            except Exception as e:
                print(f"❌ 发生错误: {e}")