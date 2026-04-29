"""
球队层面分析模块
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class TeamAnalyzer:
    def __init__(self, team_data, historical_data=None):
        self.data = team_data.copy()
        self.historical_data = historical_data or {}
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    def plot_league_standings(self, save_path=None):
        """绘制联赛积分榜柱状图"""
        fig, ax = plt.subplots(figsize=(12, 8))

        teams = self.data['球队'].head(10)
        points = self.data['积分'].head(10)

        bars = ax.barh(range(len(teams)), points, color=self.colors[0])
        ax.set_yticks(range(len(teams)))
        ax.set_yticklabels(teams)
        ax.invert_yaxis()
        ax.set_xlabel('积分', fontsize=12)
        ax.set_title('25-26赛季西甲积分榜 (前10名)', fontsize=14, fontweight='bold')

        for i, (bar, point) in enumerate(zip(bars, points)):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    str(int(point)), va='center', fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_all_teams_standings(self, save_path=None):
        """绘制全部球队积分榜"""
        fig, ax = plt.subplots(figsize=(14, 10))

        teams = self.data['球队']
        points = self.data['积分']

        sorted_idx = points.argsort()[::-1]
        teams_sorted = teams.iloc[sorted_idx]
        points_sorted = points.iloc[sorted_idx]

        colors_bar = [self.colors[i % len(self.colors)] for i in range(len(teams))]
        bars = ax.barh(range(len(teams)), points_sorted, color=colors_bar)
        ax.set_yticks(range(len(teams)))
        ax.set_yticklabels(teams_sorted, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('积分', fontsize=12)
        ax.set_title('25-26赛季西甲积分榜 (全部20支球队)', fontsize=14, fontweight='bold')
        ax.axvline(x=points_sorted.iloc[3], color='green', linestyle='--', alpha=0.5, label='欧冠区')
        ax.axvline(x=points_sorted.iloc[-3], color='red', linestyle='--', alpha=0.5, label='降级区')
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_attack_defense_radar(self, teams=None, save_path=None):
        """绘制球队攻防雷达图（已添加Min-Max标准化）"""
        if teams is None:
            teams = ['巴塞罗那', '皇家马德里', '马德里竞技', '比利亚雷亚尔']

        # 雷达图指标（根据你的Excel表头调整，确保列名完全匹配）
        metrics = ['场均进球', '场均射门', '射门转化率', '传球成功率%', '场均抢断', '场均解围']

        # 【核心修改】对所有雷达图指标进行Min-Max标准化
        from src.data_loader import DataLoader
        loader = DataLoader()
        norm_data = loader.min_max_normalize(self.data, metrics)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()

        for idx, team in enumerate(teams):
            if team not in norm_data['球队'].values:
                print(f"警告: 球队 '{team}' 不在数据中")
                continue
            team_data = norm_data[norm_data['球队'] == team].iloc[0]
            values = [team_data[metric] for metric in metrics]
            values += values[:1]  # 闭合雷达图

            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]

            ax = axes[idx]
            ax.plot(angles, values, 'o-', linewidth=2, label=team, color=self.colors[idx % len(self.colors)])
            ax.fill(angles, values, alpha=0.25, color=self.colors[idx % len(self.colors)])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=9)
            ax.set_ylim(0, 1)  # 标准化后统一范围[0,1]
            ax.set_title(f'{team}', fontsize=12, fontweight='bold', pad=20)

        # 隐藏多余子图
        for idx in range(len(teams), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('球队攻防能力雷达图对比（已标准化）', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.close()

    def plot_offensive_comparison(self, save_path=None):
        """进攻数据对比散点图"""
        fig, ax = plt.subplots(figsize=(14, 10))

        required_cols = ['场均射门', '场均进球', '积分', '射正率%']
        for col in required_cols:
            if col not in self.data.columns:
                print(f"警告: 缺少列 '{col}'")
                return

        scatter = ax.scatter(self.data['场均射门'], self.data['场均进球'],
                             s=self.data['积分'] * 3, c=self.data['射正率%'],
                             cmap='RdYlGn', alpha=0.7, edgecolors='black')

        # 标注所有球队（积分>=40或场均进球>=1.5）
        teams_to_label = self.data[(self.data['积分'] >= 1) | (self.data['场均进球'] >= 1.5)]
        for _, row in teams_to_label.iterrows():
            ax.annotate(row['球队'], (row['场均射门'], row['场均进球']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax.set_xlabel('场均射门次数', fontsize=12)
        ax.set_ylabel('场均进球数', fontsize=12)
        ax.set_title('球队进攻效率分析 (气泡大小=积分, 颜色=射正率)', fontsize=14, fontweight='bold')

        cbar = plt.colorbar(scatter)
        cbar.set_label('射正率 %', fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_defensive_heatmap(self, save_path=None):
        """防守数据热力图（已添加Min-Max标准化）"""
        defensive_cols = ['场均抢断', '场均拦截', '封堵', '场均解围', '场均黄牌', '场均红牌']
        available_cols = [col for col in defensive_cols if col in self.data.columns]
        if len(available_cols) < 3:
            print("警告: 防守数据列不足，跳过热力图")
            return

        defensive_data = self.data[['球队'] + available_cols].set_index('球队')

        # 【核心修改】使用通用Min-Max标准化方法
        from src.data_loader import DataLoader
        loader = DataLoader()
        defensive_norm = loader.min_max_normalize(defensive_data)

        plt.figure(figsize=(12, 16))
        sns.heatmap(defensive_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                    annot_kws={'size': 9}, cbar_kws={'label': '标准化值 [0,1]'})
        plt.title('球队防守能力热力图（已标准化）', fontsize=14, fontweight='bold')
        plt.xlabel('防守指标', fontsize=12)
        plt.ylabel('球队', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.close()

    def plot_passing_analysis(self, save_path=None):
        """传球能力分析"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 传球成功率分布
        ax1.hist(self.data['传球成功率%'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(self.data['传球成功率%'].mean(), color='red', linestyle='--',
                    label=f'平均值: {self.data["传球成功率%"].mean():.1f}%')
        ax1.set_xlabel('传球成功率 (%)', fontsize=12)
        ax1.set_ylabel('球队数量', fontsize=12)
        ax1.set_title('传球成功率分布', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 传球次数 vs 传球成功率
        scatter = ax2.scatter(self.data['场均传球'], self.data['传球成功率%'],
                              s=self.data['积分'] * 2, c=self.colors[0], alpha=0.6, edgecolors='black')
        ax2.set_xlabel('场均传球次数', fontsize=12)
        ax2.set_ylabel('传球成功率 (%)', fontsize=12)
        ax2.set_title('传球次数与成功率关系 (气泡大小=积分)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 标注传球强队（传球成功率前5）
        top_pass = self.data.nlargest(20, '传球成功率%')
        for _, row in top_pass.iterrows():
            ax2.annotate(row['球队'], (row['场均传球'], row['传球成功率%']),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_goals_analysis(self, save_path=None):
        """进球与失球分析"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 进球 vs 失球散点图
        self.data['净胜球'] = self.data['场均进球'] - self.data['场均失球']
        scatter = ax1.scatter(self.data['场均失球'], self.data['场均进球'],
                              s=self.data['积分'] * 3, c=self.data['排名'],
                              cmap='RdYlGn_r', alpha=0.7, edgecolors='black')
        ax1.set_xlabel('场均失球', fontsize=12)
        ax1.set_ylabel('场均进球', fontsize=12)
        ax1.set_title('攻防平衡分析 (颜色=排名, 气泡=积分)', fontsize=14, fontweight='bold')

        # 添加对角线
        max_val = max(self.data['场均进球'].max(), self.data['场均失球'].max())
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='进球=失球')
        ax1.legend()

        # 标注积分前8的球队
        top_teams = self.data.head(8)
        for _, row in top_teams.iterrows():
            ax1.annotate(row['球队'], (row['场均失球'], row['场均进球']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('排名', fontsize=10)

        # 净胜球分布
        colors_bar = ['green' if x > 0 else 'red' for x in self.data['净胜球']]
        ax2.barh(range(len(self.data)), self.data['净胜球'], color=colors_bar, alpha=0.7)
        ax2.set_yticks(range(len(self.data)))
        ax2.set_yticklabels(self.data['球队'], fontsize=8)
        ax2.set_xlabel('场均净胜球', fontsize=12)
        ax2.set_title('球队场均净胜球', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_team_style_scatter(self, save_path=None):
        """球队风格散点图"""
        fig, ax = plt.subplots(figsize=(14, 10))

        # 计算控球指数和进攻指数
        self.data['控球指数'] = self.data['传球成功率%'] * 0.5 + self.data['场均传球'] / 10 * 0.5
        self.data['进攻指数'] = self.data['场均进球'] * 10 + self.data['射门转化率']

        # 按积分着色
        scatter = ax.scatter(self.data['控球指数'], self.data['进攻指数'],
                             s=self.data['积分'] * 4, c=self.data['排名'],
                             cmap='RdYlGn_r', alpha=0.7, edgecolors='black')

        ax.set_xlabel('控球指数', fontsize=12)
        ax.set_ylabel('进攻指数', fontsize=12)
        ax.set_title('球队风格分析 (控球型 vs 进攻型)', fontsize=14, fontweight='bold')

        # 添加象限分割线
        ax.axhline(y=self.data['进攻指数'].median(), color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=self.data['控球指数'].median(), color='gray', linestyle='--', alpha=0.5)

        # 添加象限标签
        ax.text(0.95, 0.95, '控球进攻型', transform=ax.transAxes, ha='right', va='top',
                fontsize=12, fontweight='bold', color='green')
        ax.text(0.05, 0.95, '反击型', transform=ax.transAxes, ha='left', va='top',
                fontsize=12, fontweight='bold', color='orange')
        ax.text(0.95, 0.05, '控球保守型', transform=ax.transAxes, ha='right', va='bottom',
                fontsize=12, fontweight='bold', color='blue')
        ax.text(0.05, 0.05, '防守型', transform=ax.transAxes, ha='left', va='bottom',
                fontsize=12, fontweight='bold', color='red')

        # 标注所有球队
        for _, row in self.data.iterrows():
            ax.annotate(row['球队'], (row['控球指数'], row['进攻指数']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        cbar = plt.colorbar(scatter)
        cbar.set_label('排名', fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_historical_comparison(self, save_path=None):
        """历史赛季对比"""
        if not self.historical_data:
            print("警告: 没有历史数据")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 收集各赛季冠军积分
        seasons = []
        champion_points = []
        for season, df in sorted(self.historical_data.items()):
            if '积分' in df.columns and len(df) > 0:
                seasons.append(season)
                champion_points.append(df['积分'].iloc[0])

        # 添加当前赛季
        if len(self.data) > 0:
            seasons.append('25-26')
            champion_points.append(self.data['积分'].iloc[0])

        # 冠军积分趋势
        ax1.plot(seasons, champion_points, 'o-', linewidth=2, markersize=8, color=self.colors[0])
        ax1.set_xlabel('赛季', fontsize=12)
        ax1.set_ylabel('冠军积分', fontsize=12)
        ax1.set_title('西甲冠军积分趋势', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 各赛季进球数对比
        seasons_goals = []
        total_goals = []
        for season, df in sorted(self.historical_data.items()):
            if '进球数' in df.columns:
                seasons_goals.append(season)
                total_goals.append(df['进球数'].sum())

        if seasons_goals:
            ax2.bar(seasons_goals, total_goals, color=self.colors[1], alpha=0.7, edgecolor='black')
            ax2.set_xlabel('赛季', fontsize=12)
            ax2.set_ylabel('总进球数', fontsize=12)
            ax2.set_title('各赛季总进球数', fontsize=14, fontweight='bold')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_discipline_analysis(self, save_path=None):
        """纪律数据分析"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 黄牌数分布
        sorted_data = self.data.sort_values('场均黄牌', ascending=True)
        ax1.barh(range(len(sorted_data)), sorted_data['场均黄牌'], color='orange', alpha=0.7, edgecolor='black')
        ax1.set_yticks(range(len(sorted_data)))
        ax1.set_yticklabels(sorted_data['球队'], fontsize=8)
        ax1.set_xlabel('场均黄牌数', fontsize=12)
        ax1.set_title('球队场均黄牌数排名', fontsize=14, fontweight='bold')

        # 红牌数分布
        sorted_data2 = self.data.sort_values('场均红牌', ascending=True)
        colors_red = ['darkred' if x > 0.2 else 'lightcoral' for x in sorted_data2['场均红牌']]
        ax2.barh(range(len(sorted_data2)), sorted_data2['场均红牌'], color=colors_red, alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(sorted_data2)))
        ax2.set_yticklabels(sorted_data2['球队'], fontsize=8)
        ax2.set_xlabel('场均红牌数', fontsize=12)
        ax2.set_title('球队场均红牌数排名', fontsize=14, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def team_style_analysis(self):
        """球队风格分析"""
        self.data['控球指数'] = self.data['传球成功率%'] * 0.4 + self.data['场均传球'] / 100 * 0.6
        self.data['进攻指数'] = self.data['场均进球'] * 0.5 + self.data['射门转化率'] * 0.5
        self.data['防守指数'] = (self.data['场均抢断'] + self.data['场均拦截']) * 0.5 + self.data['场均解围'] * 0.5

        conditions = [
            (self.data['控球指数'] > self.data['控球指数'].median()) &
            (self.data['进攻指数'] > self.data['进攻指数'].median()),
            (self.data['控球指数'] > self.data['控球指数'].median()) &
            (self.data['进攻指数'] <= self.data['进攻指数'].median()),
            (self.data['控球指数'] <= self.data['控球指数'].median()) &
            (self.data['进攻指数'] > self.data['进攻指数'].median()),
            (self.data['控球指数'] <= self.data['控球指数'].median()) &
            (self.data['进攻指数'] <= self.data['进攻指数'].median())
        ]
        choices = ['控球进攻型', '控球保守型', '反击型', '防守型']
        self.data['战术风格'] = np.select(conditions, choices)

        return self.data[['球队', '控球指数', '进攻指数', '防守指数', '战术风格']]


if __name__ == '__main__':
    from data_loader import DataLoader

    loader = DataLoader()
    loader.load_current_season_team()
    team_data = loader.preprocess_team_data()

    analyzer = TeamAnalyzer(team_data)
    analyzer.plot_league_standings()
