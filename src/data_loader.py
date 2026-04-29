"""
数据加载、预处理与清洗模块
"""
import pandas as pd
import numpy as np
import os
import glob


class DataLoader:
    def __init__(self, data_path='../data'):
        self.data_path = data_path
        self.team_data = None          # 25-26赛季球队总数据
        self.player_data = None        # 25-26赛季球员数据
        self.historical_data = {}      # 历史赛季积分榜
        self.round_data = {}           # 逐轮比赛详细数据

    # ================= 数据加载方法 =================

    def load_current_season_team(self):
        """加载25-26赛季前29轮球队数据"""
        file_path = os.path.join(self.data_path, '25-26赛季前29轮球队数据.xlsx')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

        self.team_data = pd.read_excel(file_path)

        # 自动执行数据清洗
        self.team_data = self._clean_generic_data(self.team_data, data_type='team')

        print(f"球队数据加载与清洗完成: {len(self.team_data)} 支球队")
        return self.team_data

    def load_current_season_player(self):
        """加载25-26赛季前29轮球员数据"""
        file_path = os.path.join(self.data_path, '25-26赛季前29轮个人数据.xlsx')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

        self.player_data = pd.read_excel(file_path)

        # 统一列名，便于后续处理
        self.player_data.rename(columns={
            '期进球(xG)': '预期进球(xG)',
            '期助攻(xA)': '预期助攻(xA)',
            '场时间(分)': '出场时间(分钟)'
        }, inplace=True)

        # 自动执行数据清洗
        self.player_data = self._clean_generic_data(self.player_data, data_type='player')

        print(f"球员数据加载与清洗完成: {len(self.player_data)} 名球员")
        return self.player_data

    def load_historical_data(self):
        """加载历史赛季积分榜"""
        season_files = glob.glob(os.path.join(self.data_path, '??-??赛季.xlsx'))

        for file in season_files:
            season = os.path.basename(file).replace('.xlsx', '')
            df = pd.read_excel(file)

            # 统一历史表字段
            df.rename(columns={'阵容': '球队'}, inplace=True)

            # 自动清洗
            df = self._clean_generic_data(df, data_type='historical')

            self.historical_data[season] = df
            print(f"加载并清洗 {season}: {len(df)} 支球队")

        print(f"历史数据加载完成: {len(self.historical_data)} 个赛季")
        return self.historical_data

    def load_round_data(self):
        """加载逐轮比赛详细数据"""
        round_files = glob.glob(os.path.join(self.data_path, '25-26赛季第*轮.xlsx'))

        for file in round_files:
            filename = os.path.basename(file)
            round_num = filename.replace('25-26赛季第', '').replace('轮.xlsx', '')
            try:
                round_num = int(round_num)
            except Exception:
                pass

            df = pd.read_excel(file)

            # 统一逐轮数据客队字段命名
            df.rename(columns={
                '进球数(客)': '客队进球数',
                '控球率(客)': '客队控球率',
                '角球(客)': '客队角球',
                '射门(客)': '客队射门',
                '射正(客)': '客队射正',
                '射偏(客)': '客队射偏',
                '被封堵(客)': '客队被封堵',
                '禁区内射门(客)': '客队禁区内射门',
                '禁区外射门(客)': '客队禁区外射门',
                '传球(客)': '客队传球',
                '传球成功率(客)': '客队传球成功率',
                '门将扑救(客)': '客队门将扑救',
                '红牌(客)': '客队红牌',
                '黄牌(客)': '客队黄牌',
                '抢断(客)': '客队抢断',
                '拦截(客)': '客队拦截'
            }, inplace=True)

            # 自动清洗
            df = self._clean_generic_data(df, data_type='round')

            self.round_data[round_num] = df

        print(f"逐轮数据加载与清洗完成: {len(self.round_data)} 轮")
        return self.round_data

    # ================= 通用数据清洗模块 =================

    def _clean_generic_data(self, df, data_type='generic'):
        """
        通用数据清洗流水线
        说明：
        - 保留缺失值处理
        - 保留异常值检测
        - 不对体育统计中的极端值做截断修改，避免破坏真实表现
        """
        print(f"\n>>> 开始清洗 {data_type} 数据...")

        # 1. 完整性检查
        self._report_missing_data(df, data_type)

        # 2. 处理缺失值
        df = self._handle_missing_values(df)

        # 3. 检测异常值（只检测，不修改）
        df = self._handle_outliers(df)

        print(f">>> {data_type} 数据清洗完成。\n")
        return df

    def _report_missing_data(self, df, name):
        """打印数据缺失情况报告"""
        total = len(df)
        missing_stats = df.isnull().sum()
        missing_stats = missing_stats[missing_stats > 0]

        if missing_stats.empty:
            print(f"  ✓ [{name}] 数据完整性良好，无缺失值。")
        else:
            print(f"  ⚠️ [{name}] 发现缺失值:")
            for col, count in missing_stats.items():
                pct = (count / total) * 100 if total > 0 else 0
                print(f"    - 列 '{col}': {count} 条缺失 ({pct:.2f}%)")

    def _handle_missing_values(self, df):
        """
        智能填充缺失值
        策略:
        - 文本/分类列: 前向填充，再用 'Unknown' 填充
        - 数值列 (比率类): 均值
        - 数值列 (普通统计类): 中位数
        """
        df = df.copy()

        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_object_dtype(df[col]):
                    df[col] = df[col].ffill().fillna('Unknown')
                else:
                    if any(keyword in str(col) for keyword in ['率', '成功率', '%', '百分比']):
                        fill_val = df[col].mean()
                    else:
                        fill_val = df[col].median()

                    df[col] = df[col].fillna(fill_val)

        return df

    def _handle_outliers(self, df):
        """
        检测异常值，但不修改原始数据
        说明：
        体育数据中的高值通常是真实优秀表现，不应直接截断。
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].dropna().empty:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                continue

            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])

            if outlier_count > 0:
                print(f"  ⚠️ 列 '{col}' 检测到 {outlier_count} 个异常值候选，但未修改原始数据。")

        return df

    # ================= 预处理方法 =================

    def preprocess_team_data(self):
        """球队数据预处理（计算场均指标）"""
        if self.team_data is None:
            raise ValueError("请先加载球队数据")

        df = self.team_data.copy()

        df['场均进球'] = (df['进球'] / df['场次']).round(2)
        df['场均失球'] = (df['失球'] / df['场次']).round(2)
        df['场均射门'] = (df['射门'] / df['场次']).round(2)
        df['场均射正'] = (df['射正'] / df['场次']).round(2)
        df['射门转化率'] = (df['进球'] / df['射门'] * 100).replace([np.inf, -np.inf], np.nan).fillna(0).round(2)
        df['场均传球'] = (df['传球尝试'] / df['场次']).round(2)
        df['场均抢断'] = (df['抢断'] / df['场次']).round(2)
        df['场均拦截'] = (df['拦截'] / df['场次']).round(2)
        df['场均解围'] = (df['解围'] / df['场次']).round(2)
        df['场均角球'] = (df['角球'] / df['场次']).round(2)
        df['场均黄牌'] = (df['黄牌'] / df['场次']).round(2)
        df['场均红牌'] = (df['红牌'] / df['场次']).round(2)

        return df

    def preprocess_player_data(self):
        """球员数据预处理"""
        if self.player_data is None:
            raise ValueError("请先加载球员数据")

        df = self.player_data.copy()

        # 保留你原来的筛选逻辑
        df = df[df['出场次数'] >= 5].reset_index(drop=True)

        df['场均进球'] = (df['进球数'] / df['出场次数']).round(2)
        df['场均助攻'] = (df['助攻数'] / df['出场次数']).round(2)
        df['场均射门'] = (df['总射门次数'] / df['出场次数']).round(2)
        df['场均射正'] = (df['射正次数'] / df['出场次数']).round(2)
        df['场均时间'] = (df['出场时间(分钟)'] / df['出场次数']).round(2)

        df['进球效率'] = (
            (df['进球数'] / df['总射门次数']) * 100
        ).replace([np.inf, -np.inf], np.nan).fillna(0).round(2)

        df['xG差值'] = (df['进球数'] - df['预期进球(xG)']).round(2)

        position_map = {
            '前锋': 'FW',
            '中场': 'MF',
            '后卫': 'DF',
            '门将': 'GK'
        }
        df['位置缩写'] = df['球员位置'].map(position_map)

        return df

    def calculate_team_recent_form(self, n=5):
        """计算每支球队近n轮的状态"""
        if not self.round_data:
            raise ValueError("请先加载逐轮比赛数据")
        if self.team_data is None:
            raise ValueError("请先加载球队数据")

        all_rounds = pd.concat(self.round_data.values(), ignore_index=True)
        recent_form = {}

        for team in self.team_data['球队'].tolist():
            home_matches = all_rounds[all_rounds['主队'] == team].tail(n)
            away_matches = all_rounds[all_rounds['客队'] == team].tail(n)

            home_goals = home_matches['进球数'].sum() if '进球数' in home_matches.columns else 0
            away_goals = away_matches['客队进球数'].sum() if '客队进球数' in away_matches.columns else 0

            home_conceded = home_matches['客队进球数'].sum() if '客队进球数' in home_matches.columns else 0
            away_conceded = away_matches['进球数'].sum() if '进球数' in away_matches.columns else 0

            total_goals = home_goals + away_goals
            total_conceded = home_conceded + away_conceded

            matches_played = len(home_matches) + len(away_matches)

            if matches_played == 0:
                recent_form[team] = {
                    '近5轮场均进球': 0,
                    '近5轮场均失球': 0
                }
            else:
                recent_form[team] = {
                    '近5轮场均进球': round(total_goals / matches_played, 2),
                    '近5轮场均失球': round(total_conceded / matches_played, 2)
                }

        return pd.DataFrame.from_dict(recent_form, orient='index').reset_index().rename(columns={'index': '球队'})

    def min_max_normalize(self, df, columns=None):
        """
        Min-Max标准化，将指定列的数据映射到[0,1]区间
        公式：X' = (X - Xmin) / (Xmax - Xmin)
        """
        df_norm = df.copy()

        if columns is None:
            columns = df_norm.select_dtypes(include=[np.number]).columns

        for col in columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()

            if max_val == min_val:
                df_norm[col] = 0.5
            else:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

        return df_norm