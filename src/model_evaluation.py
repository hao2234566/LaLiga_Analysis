"""
模型评估与优化模块
包含：准确率、精确率、召回率、F1分数、AUC、混淆矩阵、特征重要性分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, log_loss
)
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    """
    模型评估器：多指标综合评价 + 特征重要性分析
    """

    def __init__(self, team_data, round_data):
        self.team_data = team_data.copy()
        self.round_data = round_data.copy() if round_data else {}
        self.models = {}
        self.evaluation_results = {}
        self.feature_importance = {}
        # 【核心修改】替换为论文要求的Min-Max标准化，映射到[0,1]区间
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_match_features(self):
        """
        从逐轮比赛数据中提取特征，构建训练数据集
        特征包括：球队历史 stats + 近期状态 + 主客场因素
        """
        if not self.round_data:
            print("❌ 缺少逐轮数据，无法构建训练集")
            return None, None

        all_matches = pd.concat(self.round_data.values(), ignore_index=True)

        # 构建球队统计字典
        team_stats = {}
        for _, team in self.team_data.iterrows():
            team_stats[team['球队']] = {
                '场均进球': team.get('场均进球', team['进球'] / team['场次']),
                '场均失球': team.get('场均失球', team['失球'] / team['场次']),
                '场均射门': team.get('场均射门', team['射门'] / team['场次']),
                '场均射正': team.get('场均射正', team['射正'] / team['场次']),
                '射门转化率': team.get('射门转化率', team['进球'] / max(team['射门'], 1) * 100),
                '传球成功率': team.get('传球成功率', team['传球成功'] / max(team['传球尝试'], 1) * 100),
                '场均抢断': team.get('场均抢断', team['抢断'] / team['场次']),
                '场均拦截': team.get('场均拦截', team['拦截'] / team['场次']),
                '场均角球': team.get('场均角球', team['角球'] / team['场次']),
                '场均黄牌': team.get('场均黄牌', team['黄牌'] / team['场次']),
                '控球率': team.get('控球率', 50),
                '积分': team['积分'],
                '排名': team.get('排名', 0)
            }

        features_list = []
        labels_list = []
        match_info_list = []

        for _, match in all_matches.iterrows():
            home_team = match['主队']
            away_team = match['客队']
            home_goals = match['进球数']
            away_goals = match['客队进球']

            if home_team not in team_stats or away_team not in team_stats:
                continue

            home = team_stats[home_team]
            away = team_stats[away_team]

            # 构建特征向量
            feature = {
                # 主队特征
                'home_avg_goals': home['场均进球'],
                'home_avg_conceded': home['场均失球'],
                'home_shots': home['场均射门'],
                'home_shots_on_target': home['场均射正'],
                'home_conversion': home['射门转化率'],
                'home_pass_accuracy': home['传球成功率'],
                'home_tackles': home['场均抢断'],
                'home_interceptions': home['场均拦截'],
                'home_corners': home['场均角球'],
                'home_yellow': home['场均黄牌'],
                'home_possession': home['控球率'],
                'home_points': home['积分'],
                'home_rank': home['排名'],

                # 客队特征
                'away_avg_goals': away['场均进球'],
                'away_avg_conceded': away['场均失球'],
                'away_shots': away['场均射门'],
                'away_shots_on_target': away['场均射正'],
                'away_conversion': away['射门转化率'],
                'away_pass_accuracy': away['传球成功率'],
                'away_tackles': away['场均抢断'],
                'away_interceptions': away['场均拦截'],
                'away_corners': away['场均角球'],
                'away_yellow': away['场均黄牌'],
                'away_possession': away['控球率'],
                'away_points': away['积分'],
                'away_rank': away['排名'],

                # 衍生特征
                'points_diff': home['积分'] - away['积分'],
                'rank_diff': away['排名'] - home['排名'],
                'goals_diff': home['场均进球'] - away['场均进球'],
                'defense_diff': away['场均失球'] - home['场均失球'],
                'form_diff': (home['场均进球'] - home['场均失球']) - (away['场均进球'] - away['场均失球']),

                # 主客场优势
                'home_advantage': 1
            }

            features_list.append(feature)

            # 标签：0=客胜, 1=平局, 2=主胜
            if home_goals > away_goals:
                labels_list.append(2)
            elif home_goals == away_goals:
                labels_list.append(1)
            else:
                labels_list.append(0)

            match_info_list.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_goals': home_goals,
                'away_goals': away_goals
            })

        X = pd.DataFrame(features_list)
        y = np.array(labels_list)
        match_info = pd.DataFrame(match_info_list)

        print(f"✅ 构建训练集完成: {len(X)} 场比赛, {X.shape[1]} 个特征")
        print(f"   主胜: {sum(y == 2)}场, 平局: {sum(y == 1)}场, 客胜: {sum(y == 0)}场")

        return X, y, match_info

    def train_and_evaluate(self, X, y, test_size=0.2, random_state=42):
        """
        训练多个模型并进行综合评估
        """
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 定义要训练的模型
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=random_state,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced',
                multi_class='ovr'
            )
        }

        results = {}

        for name, model in models.items():
            print(f"\n{'=' * 50}")
            print(f"训练模型: {name}")
            print('=' * 50)

            # 根据模型类型选择是否使用标准化数据
            # 所有模型统一使用标准化后的数据（完全符合论文数据预处理要求）
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)
            self.models[name] = (model, True)

            # 计算评估指标
            metrics = self._calculate_metrics(y_test, y_pred, y_prob)
            results[name] = metrics

            # 打印分类报告
            target_names = ['客胜', '平局', '主胜']
            print("\n分类报告:")
            print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

            # 保存特征重要性
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            elif hasattr(model, 'coef_'):
                coef_avg = np.abs(model.coef_).mean(axis=0)
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X.columns,
                    'importance': coef_avg
                }).sort_values('importance', ascending=False)

        self.evaluation_results = results
        self.X_columns = X.columns.tolist()

        return results, X_test, y_test

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """
        计算多分类问题的各项评估指标
        """
        metrics = {}

        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # 多分类精确率、召回率、F1 (使用weighted平均)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # 每个类别的详细指标
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

        # AUC
        try:
            y_true_bin = np.zeros((len(y_true), 3))
            for i, label in enumerate(y_true):
                y_true_bin[i, label] = 1
            metrics['auc_ovr'] = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='weighted')
        except:
            metrics['auc_ovr'] = None

        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # 对数损失
        metrics['log_loss'] = log_loss(y_true, y_prob)

        return metrics

    def cross_validate(self, X, y, cv=5):
        """
        交叉验证评估模型稳定性
        """
        print(f"\n{'=' * 50}")
        print(f"开始 {cv} 折交叉验证")
        print('=' * 50)

        cv_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        for name, (model, needs_scaling) in self.models.items():
            if needs_scaling:
                X_scaled = self.scaler.fit_transform(X)
                scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
            else:
                scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

            cv_results[name] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return cv_results

    def plot_evaluation_metrics(self, save_path=None):
        """
        绘制综合评估指标对比图
        """
        if not self.evaluation_results:
            print("❌ 请先运行 train_and_evaluate()")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        models = list(self.evaluation_results.keys())
        colors = ['#2E86AB', '#A23B72', '#F18F01']

        # 1. 基础指标对比
        ax1 = axes[0, 0]
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        x = np.arange(len(metrics))
        width = 0.25

        for i, model in enumerate(models):
            values = [self.evaluation_results[model][m] for m in metrics]
            ax1.bar(x + i * width, values, width, label=model, color=colors[i], alpha=0.8)

        ax1.set_ylabel('分数', fontsize=11)
        ax1.set_title('综合指标对比 (Weighted)', fontsize=12, fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(['准确率', '精确率', '召回率', 'F1分数'], fontsize=10)
        ax1.legend(loc='lower right')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)

        # 2. 各类别精确率
        ax2 = axes[0, 1]
        classes = ['客胜', '平局', '主胜']
        x = np.arange(len(classes))

        for i, model in enumerate(models):
            values = self.evaluation_results[model]['precision_per_class']
            ax2.bar(x + i * width, values, width, label=model, color=colors[i], alpha=0.8)

        ax2.set_ylabel('精确率', fontsize=11)
        ax2.set_title('各类别精确率对比', fontsize=12, fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(classes, fontsize=10)
        ax2.legend()
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)

        # 3. 各类别召回率
        ax3 = axes[0, 2]
        for i, model in enumerate(models):
            values = self.evaluation_results[model]['recall_per_class']
            ax3.bar(x + i * width, values, width, label=model, color=colors[i], alpha=0.8)

        ax3.set_ylabel('召回率', fontsize=11)
        ax3.set_title('各类别召回率对比', fontsize=12, fontweight='bold')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(classes, fontsize=10)
        ax3.legend()
        ax3.set_ylim(0, 1)
        ax3.grid(axis='y', alpha=0.3)

        # 4. 混淆矩阵 (以RandomForest为例)
        ax4 = axes[1, 0]
        cm = self.evaluation_results['RandomForest']['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                    xticklabels=['客胜', '平局', '主胜'],
                    yticklabels=['客胜', '平局', '主胜'])
        ax4.set_title('混淆矩阵 (RandomForest)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('预测标签', fontsize=10)
        ax4.set_ylabel('真实标签', fontsize=10)

        # 5. 模型综合得分雷达图
        ax5 = axes[1, 1]
        categories = ['准确率', '精确率\n(Macro)', '召回率\n(Macro)', 'F1\n(Macro)']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        for i, model in enumerate(models):
            values = [
                self.evaluation_results[model]['accuracy'],
                self.evaluation_results[model]['precision_macro'],
                self.evaluation_results[model]['recall_macro'],
                self.evaluation_results[model]['f1_macro']
            ]
            values += values[:1]
            ax5.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax5.fill(angles, values, alpha=0.15, color=colors[i])

        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories, fontsize=9)
        ax5.set_ylim(0, 1)
        ax5.set_title('模型性能雷达图', fontsize=12, fontweight='bold')
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax5.grid(True)

        # 6. 指标汇总表格
        ax6 = axes[1, 2]
        ax6.axis('off')

        table_data = []
        for model in models:
            r = self.evaluation_results[model]
            table_data.append([
                model,
                f"{r['accuracy']:.3f}",
                f"{r['precision_weighted']:.3f}",
                f"{r['recall_weighted']:.3f}",
                f"{r['f1_weighted']:.3f}",
                f"{r['log_loss']:.3f}"
            ])

        table = ax6.table(
            cellText=table_data,
            colLabels=['模型', '准确率', '精确率', '召回率', 'F1分数', '对数损失'],
            loc='center',
            cellLoc='center',
            colColours=['#4472C4'] * 6,
            colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)

        for i in range(6):
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax6.set_title('评估指标汇总', fontsize=12, fontweight='bold', y=0.95)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 评估指标图已保存: {save_path}")

        plt.show()
        return fig

    def plot_feature_importance(self, top_n=15, save_path=None):
        """
        绘制特征重要性分析图
        """
        if not self.feature_importance:
            print("❌ 请先运行 train_and_evaluate()")
            return

        n_models = len(self.feature_importance)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 10))

        if n_models == 1:
            axes = [axes]

        colors_map = {'RandomForest': '#2E86AB', 'GradientBoosting': '#A23B72', 'LogisticRegression': '#F18F01'}

        # 特征名中文映射
        feature_names_cn = {
            'home_avg_goals': '主队场均进球',
            'home_avg_conceded': '主队场均失球',
            'home_shots': '主队场均射门',
            'home_shots_on_target': '主队场均射正',
            'home_conversion': '主队射门转化率',
            'home_pass_accuracy': '主队传球成功率',
            'home_tackles': '主队场均抢断',
            'home_interceptions': '主队场均拦截',
            'home_corners': '主队场均角球',
            'home_yellow': '主队场均黄牌',
            'home_possession': '主队控球率',
            'home_points': '主队积分',
            'home_rank': '主队排名',
            'away_avg_goals': '客队场均进球',
            'away_avg_conceded': '客队场均失球',
            'away_shots': '客队场均射门',
            'away_shots_on_target': '客队场均射正',
            'away_conversion': '客队射门转化率',
            'away_pass_accuracy': '客队传球成功率',
            'away_tackles': '客队场均抢断',
            'away_interceptions': '客队场均拦截',
            'away_corners': '客队场均角球',
            'away_yellow': '客队场均黄牌',
            'away_possession': '客队控球率',
            'away_points': '客队积分',
            'away_rank': '客队排名',
            'points_diff': '积分差',
            'rank_diff': '排名差',
            'goals_diff': '进球差',
            'defense_diff': '防守差',
            'form_diff': '状态差',
            'home_advantage': '主场优势'
        }

        for idx, (model_name, importance_df) in enumerate(self.feature_importance.items()):
            ax = axes[idx]

            # 取前N个重要特征
            top_features = importance_df.head(top_n).sort_values('importance', ascending=True)
            top_features['feature_cn'] = top_features['feature'].map(feature_names_cn)

            color = colors_map.get(model_name, '#2E86AB')
            ax.barh(top_features['feature_cn'], top_features['importance'], color=color, alpha=0.8)
            ax.set_xlabel('重要性', fontsize=11)
            ax.set_title(f'{model_name}\n特征重要性 (Top {top_n})', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            # 添加数值标签
            for i, (idx_row, row) in enumerate(top_features.iterrows()):
                ax.text(row['importance'] + 0.005, i, f'{row["importance"]:.3f}',
                        va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 特征重要性图已保存: {save_path}")

        plt.show()
        return fig

    def plot_confusion_matrices(self, save_path=None):
        """
        绘制所有模型的混淆矩阵对比
        """
        if not self.evaluation_results:
            print("❌ 请先运行 train_and_evaluate()")
            return

        models = list(self.evaluation_results.keys())
        n_models = len(models)

        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        for idx, model in enumerate(models):
            cm = self.evaluation_results[model]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=['客胜', '平局', '主胜'],
                        yticklabels=['客胜', '平局', '主胜'],
                        cbar_kws={'label': '样本数'})
            axes[idx].set_title(f'{model}\n混淆矩阵', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('预测标签', fontsize=10)
            axes[idx].set_ylabel('真实标签', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 混淆矩阵对比图已保存: {save_path}")

        plt.show()
        return fig

    def get_feature_importance_summary(self):
        """
        获取特征重要性汇总分析
        """
        if not self.feature_importance:
            return None

        summary = []

        for model_name, importance_df in self.feature_importance.items():
            top5 = importance_df.head(5)
            summary.append(f"\n【{model_name}】Top 5 关键特征:")
            for idx, row in top5.iterrows():
                summary.append(f"  {idx + 1}. {row['feature']}: {row['importance']:.4f}")

        # 综合排名 (取平均)
        if len(self.feature_importance) > 1:
            all_importance = pd.DataFrame()
            for model_name, importance_df in self.feature_importance.items():
                temp = importance_df.copy()
                temp['model'] = model_name
                all_importance = pd.concat([all_importance, temp])

            avg_importance = all_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)

            summary.append("\n【综合排名】各模型平均重要性 Top 10:")
            for idx, (feature, importance) in enumerate(avg_importance.head(10).items()):
                summary.append(f"  {idx + 1}. {feature}: {importance:.4f}")

        return '\n'.join(summary)

    def backtest_prediction(self, X, y, match_info, n_recent=50):
        """
        回测验证：用最近的比赛验证模型预测效果
        """
        print(f"\n{'=' * 50}")
        print(f"回测验证 (最近 {n_recent} 场比赛)")
        print('=' * 50)

        # 取最近的比赛
        X_recent = X.tail(n_recent)
        y_recent = y[-n_recent:]
        match_recent = match_info.tail(n_recent)

        results = []

        for name, (model, needs_scaling) in self.models.items():
            if needs_scaling:
                X_scaled = self.scaler.transform(X_recent)
                y_pred = model.predict(X_scaled)
            else:
                y_pred = model.predict(X_recent)

            correct = sum(y_pred == y_recent)
            accuracy = correct / len(y_recent)

            predictions = []
            for i, (idx, match) in enumerate(match_recent.iterrows()):
                pred_label = y_pred[i]
                true_label = y_recent[i]
                pred_result = ['客胜', '平局', '主胜'][pred_label]
                true_result = ['客胜', '平局', '主胜'][true_label]
                correct_flag = '✓' if pred_label == true_label else '✗'

                predictions.append({
                    'match': f"{match['home_team']} vs {match['away_team']}",
                    'score': f"{match['home_goals']}-{match['away_goals']}",
                    'predicted': pred_result,
                    'actual': true_result,
                    'correct': correct_flag
                })

            results.append({
                'model': name,
                'accuracy': accuracy,
                'correct_count': correct,
                'total': len(y_recent),
                'predictions': predictions
            })

            print(f"\n{name}: {correct}/{len(y_recent)} 正确, 准确率: {accuracy:.2%}")

        return results


def run_model_evaluation(team_data, round_data, output_dir='output'):
    """
    运行完整的模型评估流程
    """
    print("\n" + "=" * 60)
    print("开始模型评估与特征重要性分析")
    print("=" * 60)

    # 初始化评估器
    evaluator = ModelEvaluator(team_data, round_data)

    # 1. 准备数据
    X, y, match_info = evaluator.prepare_match_features()
    if X is None:
        return None

    # 2. 训练并评估
    results, X_test, y_test = evaluator.train_and_evaluate(X, y)

    # 3. 交叉验证
    cv_results = evaluator.cross_validate(X, y)

    # 4. 绘制评估图表
    evaluator.plot_evaluation_metrics(save_path=f'{output_dir}/24_模型评估指标.png')
    evaluator.plot_feature_importance(save_path=f'{output_dir}/25_特征重要性分析.png')
    evaluator.plot_confusion_matrices(save_path=f'{output_dir}/26_混淆矩阵对比.png')

    # 5. 回测验证
    backtest_results = evaluator.backtest_prediction(X, y, match_info, n_recent=50)

    # 6. 打印特征重要性汇总
    print("\n" + "=" * 60)
    print("特征重要性分析结果")
    print("=" * 60)
    print(evaluator.get_feature_importance_summary())

    # 7. 保存详细结果到Excel
    with pd.ExcelWriter(f'{output_dir}/模型评估结果.xlsx', engine='openpyxl') as writer:
        # 各模型评估指标
        metrics_df = pd.DataFrame({
            model: {
                '准确率': r['accuracy'],
                '精确率(Macro)': r['precision_macro'],
                '召回率(Macro)': r['recall_macro'],
                'F1分数(Macro)': r['f1_macro'],
                '精确率(Weighted)': r['precision_weighted'],
                '召回率(Weighted)': r['recall_weighted'],
                'F1分数(Weighted)': r['f1_weighted'],
                '对数损失': r['log_loss']
            }
            for model, r in results.items()
        }).T
        metrics_df.to_excel(writer, sheet_name='评估指标汇总')

        # 各类别详细指标
        for model, r in results.items():
            class_metrics = pd.DataFrame({
                '客胜': [r['precision_per_class'][0], r['recall_per_class'][0], r['f1_per_class'][0]],
                '平局': [r['precision_per_class'][1], r['recall_per_class'][1], r['f1_per_class'][1]],
                '主胜': [r['precision_per_class'][2], r['recall_per_class'][2], r['f1_per_class'][2]]
            }, index=['精确率', '召回率', 'F1分数'])
            class_metrics.to_excel(writer, sheet_name=f'{model}_各类别指标')

        # 特征重要性
        for model, importance_df in evaluator.feature_importance.items():
            importance_df.to_excel(writer, sheet_name=f'{model}_特征重要性', index=False)

        # 交叉验证结果
        cv_df = pd.DataFrame({
            model: [r['mean'], r['std']] + r['scores'].tolist()
            for model, r in cv_results.items()
        }, index=['均值', '标准差'] + [f'折{i + 1}' for i in range(len(list(cv_results.values())[0]['scores']))])
        cv_df.to_excel(writer, sheet_name='交叉验证结果')

    print(f"\n✅ 详细评估结果已保存至: {output_dir}/模型评估结果.xlsx")

    return evaluator, results, cv_results
