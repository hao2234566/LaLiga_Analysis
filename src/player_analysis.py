"""
球员层面分析模块
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class PlayerAnalyzer:
    def __init__(self, player_data):
        self.data = player_data.copy()
        self.position_colors = {
            'FW': '#e74c3c',
            'MF': '#3498db',
            'DF': '#2ecc71',
            'GK': '#f39c12'
        }
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    def plot_top_scorers(self, n=15, save_path=None):
        """射手榜"""
        top_scorers = self.data.nlargest(n, '进球数')

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = [self.position_colors.get(pos, 'gray') for pos in top_scorers['位置缩写']]

        bars = ax.barh(range(len(top_scorers)), top_scorers['进球数'], color=colors)
        ax.set_yticks(range(len(top_scorers)))
        ax.set_yticklabels(top_scorers['球员姓名'])
        ax.invert_yaxis()
        ax.set_xlabel('进球数', fontsize=12)
        ax.set_title(f'25-26赛季射手榜 (前{n}名)', fontsize=14, fontweight='bold')

        for bar, val in zip(bars, top_scorers['进球数']):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    str(int(val)), va='center', fontsize=10)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=pos)
                           for pos, color in self.position_colors.items()]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_top_assists(self, n=15, save_path=None):
        """助攻榜"""
        top_assists = self.data.nlargest(n, '助攻数')

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = [self.position_colors.get(pos, 'gray') for pos in top_assists['位置缩写']]

        bars = ax.barh(range(len(top_assists)), top_assists['助攻数'], color=colors)
        ax.set_yticks(range(len(top_assists)))
        ax.set_yticklabels(top_assists['球员姓名'])
        ax.invert_yaxis()
        ax.set_xlabel('助攻数', fontsize=12)
        ax.set_title(f'25-26赛季助攻榜 (前{n}名)', fontsize=14, fontweight='bold')

        for bar, val in zip(bars, top_assists['助攻数']):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    str(int(val)), va='center', fontsize=10)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=pos)
                           for pos, color in self.position_colors.items()]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_goal_assist_scatter(self, save_path=None):
        """进球-助攻散点图"""
        fig, ax = plt.subplots(figsize=(14, 10))

        df = self.data[self.data['出场次数'] >= 10]

        for pos in ['FW', 'MF', 'DF']:
            pos_data = df[df['位置缩写'] == pos]
            ax.scatter(pos_data['助攻数'], pos_data['进球数'],
                       s=pos_data['出场时间(分钟)'] / 10,
                       c=self.position_colors[pos], alpha=0.6,
                       label=pos, edgecolors='black')

        ax.set_xlabel('助攻数', fontsize=12)
        ax.set_ylabel('进球数', fontsize=12)
        ax.set_title('球员进球-助攻能力分布 (气泡大小=出场时间)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 标注进球>=8或助攻>=6的球员
        top_players = df[(df['进球数'] >= 8) | (df['助攻数'] >= 6)]
        for _, player in top_players.iterrows():
            ax.annotate(player['球员姓名'],
                        (player['助攻数'], player['进球数']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_xg_analysis(self, save_path=None):
        """预期进球(xG)分析"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        df = self.data[self.data['出场次数'] >= 10]
        ax1.scatter(df['预期进球(xG)'], df['进球数'], alpha=0.6, c='steelblue')
        ax1.plot([0, df['预期进球(xG)'].max()], [0, df['预期进球(xG)'].max()],
                 'r--', label='xG = 实际进球')
        ax1.set_xlabel('预期进球 (xG)', fontsize=12)
        ax1.set_ylabel('实际进球', fontsize=12)
        ax1.set_title('预期进球 vs 实际进球', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 标注xG>=6的球员
        top_xg = df[df['预期进球(xG)'] >= 6]
        for _, player in top_xg.iterrows():
            ax1.annotate(player['球员姓名'],
                         (player['预期进球(xG)'], player['进球数']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 进球效率排名
        df['进球效率'] = df['进球数'] / df['预期进球(xG)']
        df_eff = df[df['预期进球(xG)'] >= 3].nlargest(10, '进球效率')

        ax2.barh(range(len(df_eff)), df_eff['进球效率'], color='coral')
        ax2.set_yticks(range(len(df_eff)))
        ax2.set_yticklabels(df_eff['球员姓名'])
        ax2.invert_yaxis()
        ax2.set_xlabel('进球效率 (实际/xG)', fontsize=12)
        ax2.set_title('进球效率排名 (xG≥3)', fontsize=12, fontweight='bold')
        ax2.axvline(x=1, color='red', linestyle='--', label='效率=1')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_rating_distribution(self, save_path=None):
        """球员评分分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 评分整体分布
        ax1.hist(self.data['球员评分'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(self.data['球员评分'].mean(), color='red', linestyle='--',
                    label=f'平均值: {self.data["球员评分"].mean():.2f}')
        ax1.set_xlabel('球员评分', fontsize=12)
        ax1.set_ylabel('人数', fontsize=12)
        ax1.set_title('球员评分分布', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 各位置评分对比
        positions = ['FW', 'MF', 'DF', 'GK']
        ratings_by_pos = [self.data[self.data['位置缩写'] == pos]['球员评分'].dropna() for pos in positions]
        bp = ax2.boxplot(ratings_by_pos, labels=['前锋', '中场', '后卫', '门将'])
        ax2.set_ylabel('球员评分', fontsize=12)
        ax2.set_title('各位置球员评分分布', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_top_rated_players(self, n=15, save_path=None):
        """评分最高球员"""
        top_rated = self.data.nlargest(n, '球员评分')

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = [self.position_colors.get(pos, 'gray') for pos in top_rated['位置缩写']]

        bars = ax.barh(range(len(top_rated)), top_rated['球员评分'], color=colors)
        ax.set_yticks(range(len(top_rated)))
        ax.set_yticklabels(top_rated['球员姓名'])
        ax.invert_yaxis()
        ax.set_xlabel('球员评分', fontsize=12)
        ax.set_title(f'25-26赛季评分最高球员 (前{n}名)', fontsize=14, fontweight='bold')
        ax.set_xlim([6, 8])

        for bar, val in zip(bars, top_rated['球员评分']):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', fontsize=10)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=pos)
                           for pos, color in self.position_colors.items()]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_shooting_efficiency(self, save_path=None):
        """射门效率分析"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        df = self.data[self.data['出场次数'] >= 10]

        # 射门次数 vs 进球数
        scatter = ax1.scatter(df['总射门次数'], df['进球数'],
                              c=df['球员评分'], cmap='RdYlGn',
                              alpha=0.6, edgecolors='black')
        ax1.set_xlabel('总射门次数', fontsize=12)
        ax1.set_ylabel('进球数', fontsize=12)
        ax1.set_title('射门次数与进球数关系 (颜色=评分)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='评分')

        # 标注进球>=8的球员
        top_shooters = df[df['进球数'] >= 8]
        for _, player in top_shooters.iterrows():
            ax1.annotate(player['球员姓名'],
                         (player['总射门次数'], player['进球数']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 进球效率分布
        ax2.hist(df['进球效率'], bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(df['进球效率'].mean(), color='red', linestyle='--',
                    label=f'平均值: {df["进球效率"].mean():.1f}%')
        ax2.set_xlabel('进球效率 (%)', fontsize=12)
        ax2.set_ylabel('人数', fontsize=12)
        ax2.set_title('球员进球效率分布', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_appearance_analysis(self, save_path=None):
        """出场时间分析"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 出场时间分布
        ax1.hist(self.data['出场时间(分钟)'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax1.axvline(self.data['出场时间(分钟)'].mean(), color='red', linestyle='--',
                    label=f'平均值: {self.data["出场时间(分钟)"].mean():.0f}分钟')
        ax1.set_xlabel('出场时间 (分钟)', fontsize=12)
        ax1.set_ylabel('人数', fontsize=12)
        ax1.set_title('球员出场时间分布', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 出场次数分布
        appearances = self.data['出场次数'].value_counts().sort_index()
        ax2.bar(appearances.index, appearances.values, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('出场次数', fontsize=12)
        ax2.set_ylabel('人数', fontsize=12)
        ax2.set_title('球员出场次数分布', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_position_comparison_boxplot(self, save_path=None):
        """不同位置球员指标对比箱线图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        metrics = [
            ('球员评分', '球员评分'),
            ('场均进球', '场均进球'),
            ('场均时间', '场均时间'),
            ('进球效率', '进球效率')
        ]

        for idx, (title, col) in enumerate(metrics):
            ax = axes[idx]
            positions = ['FW', 'MF', 'DF']

            if col not in self.data.columns:
                print(f"警告: 列 '{col}' 不存在，跳过该图表")
                ax.set_title(f'{title}分布 (数据缺失)', fontsize=12)
                continue

            data_to_plot = [self.data[self.data['位置缩写'] == pos][col].dropna()
                            for pos in positions]

            bp = ax.boxplot(data_to_plot, labels=['前锋', '中场', '后卫'])
            ax.set_title(f'{title}分布', fontsize=12, fontweight='bold')
            ax.set_ylabel(title, fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.suptitle('不同位置球员能力对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.show()

    def plot_player_radar(self, player_names=None, save_path=None, metrics=None):
        """单个球员能力雷达图（已添加Min-Max标准化）"""
        if player_names is None:
            player_names = ['基利安·姆巴佩', '拉明·亚马尔', '祖德·贝林厄姆', '佩德里', '朱尔·孔德', '罗纳德·阿劳霍']
        if metrics is None:
            metrics = ['进球数', '助攻数', '预期进球(xG)', '预期助攻(xA)', '球员评分', '总射门次数', '射正次数',
                       '抢断次数', '拦截次数']

        # 过滤有效球员
        valid_players = [name for name in player_names if name in self.data['球员姓名'].values]
        if not valid_players:
            print("没有有效的球员可绘制")
            return

        # 【核心修改】对所有雷达图指标进行Min-Max标准化
        from src.data_loader import DataLoader
        loader = DataLoader()
        norm_data = loader.min_max_normalize(self.data, metrics)

        n_players = len(valid_players)
        n_cols = 3
        n_rows = (n_players + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows),
                                 subplot_kw=dict(projection='polar'))
        axes = axes.flatten() if n_players > 1 else [axes]

        for idx, player_name in enumerate(valid_players):
            player_data = norm_data[norm_data['球员姓名'] == player_name].iloc[0]
            original_data = self.data[self.data['球员姓名'] == player_name].iloc[0]
            values = [player_data[metric] for metric in metrics]
            values += values[:1]

            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]

            ax = axes[idx]
            ax.plot(angles, values, 'o-', linewidth=2, color=self.colors[idx % len(self.colors)])
            ax.fill(angles, values, alpha=0.25, color=self.colors[idx % len(self.colors)])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=9)
            ax.set_ylim(0, 1)  # 标准化后统一范围[0,1]
            position = original_data.get('位置缩写', '未知')
            ax.set_title(f'{player_name} ({position})', fontsize=12, fontweight='bold', pad=20)

        # 隐藏多余子图
        for idx in range(n_players, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('球员能力雷达图（已标准化）', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        plt.close()

    def get_best_xi(self):
        """评选最佳阵容"""
        best_xi = {}

        gk = self.data[self.data['位置缩写'] == 'GK'].nlargest(1, '球员评分')
        best_xi['GK'] = gk

        df_pos = self.data[self.data['位置缩写'] == 'DF']
        best_xi['DF'] = df_pos.nlargest(4, '球员评分')

        mf = self.data[self.data['位置缩写'] == 'MF']
        best_xi['MF'] = mf.nlargest(3, '球员评分')

        fw = self.data[self.data['位置缩写'] == 'FW']
        best_xi['FW'] = fw.nlargest(3, '球员评分')

        return best_xi


if __name__ == '__main__':
    from data_loader import DataLoader

    loader = DataLoader()
    loader.load_current_season_player()
    player_data = loader.preprocess_player_data()

    analyzer = PlayerAnalyzer(player_data)
    analyzer.plot_top_scorers()
