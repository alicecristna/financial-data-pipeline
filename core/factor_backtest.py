"""
Factor Backtest Module
Date: 2026-04-24
Description: Vectorized factor calculation and strategy backtesting
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Callable
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import spearmanr
class FactorBacktest:
    """
    向量化因子回测框架
    - 因子计算（动量、波动率、均线交叉等）
    - 信号生成（多空阈值）
    - 策略收益计算
    - 回测绩效评估
    """
    
    def __init__(self, prices: pd.DataFrame):
        """
        parameters:
        prices: DataFrame, 行=日期, 列=资产, 值=收盘价

        固定资产池策略：缺失价格不会被前向填充，也不会在组合中被静默重新
        加权。第一行收益率允许缺失（pct_change 的固有行为），之后任何缺失
        或非有限收益率都会抛出 ValueError。
        """
        self.prices = prices
        # 简单收益率：用于持仓盈亏和等权组合。不能用对数收益率求平均，
        # 那不是等权组合的收益。fill_method=None 禁止前向填充，并保证
        # pandas 2.x 与 3.x 行为一致。
        self.returns = self._validate_fixed_universe(
            prices.pct_change(fill_method=None)
        )
        self.signals = None
        self.strategy_returns = None

    @staticmethod
    def _validate_fixed_universe(
        returns: pd.DataFrame, label: str = "asset returns"
    ) -> pd.DataFrame:
        """校验固定资产池：只有第一行允许缺失，其余必须全部有限。

        缺失一个资产就在等权组合里跳过它，等于静默改变权重；这里选择直接
        报错，而不是让调用方拿到误导性的固定 1/N 结果。
        """
        if len(returns) <= 1:
            return returns

        tail = returns.iloc[1:]
        finite = np.isfinite(tail)
        if finite.to_numpy().all():
            return returns

        bad_assets = tail.columns[~finite.all(axis=0)].tolist()
        bad_rows = tail.index[~finite.all(axis=1)].tolist()[:5]
        raise ValueError(
            f"{label}: missing or non-finite value(s) after the first row; "
            f"FactorBacktest enforces a fixed universe and does not "
            f"forward-fill prices or silently reweight. "
            f"assets={bad_assets}, rows={list(bad_rows)}"
        )
        
    def momentum_factor(self, window: int = 20) -> pd.DataFrame:
        """
        计算动量因子
        window: 回溯窗口（交易日）
        返回: DataFrame，值 = (当前价 / N日前价) - 1
        """
        return self.prices / self.prices.shift(window) - 1
    
    def volatility_factor(self, window: int = 20) -> pd.DataFrame:
        """
        计算波动率因子
        返回: 年化波动率
        """
        return self.returns.rolling(window).std(ddof=0) * np.sqrt(252)
    
    def ma_crossover_factor(self, fast: int = 5, slow: int = 20) -> pd.DataFrame:
        """
        均线交叉因子
        返回: 快线 / 慢线 - 1（正值表示快线在上）
        """
        ma_fast = self.prices.rolling(fast).mean()
        ma_slow = self.prices.rolling(slow).mean()
        return ma_fast / ma_slow - 1
    
    def generate_signals(self, factor: pd.DataFrame, 
                         long_threshold: float = 0.0, 
                         short_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        根据因子值生成交易信号
        factor: 因子值 DataFrame
        long_threshold: 做多阈值（因子大于此值做多）
        short_threshold: 做空阈值（因子小于此值做空），None 表示不做空
        """
        conditions = [factor > long_threshold]
        choices = [1]

        if short_threshold is not None:
            conditions.append(factor < short_threshold)
            choices.append(-1)

        self.signals = pd.DataFrame(
            np.select(conditions, choices, default=0),
            index=factor.index,
            columns=factor.columns
        )
        return self.signals
    
    def calculate_strategy_returns(self, signals: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        计算策略收益率
        策略收益 = 信号(t-1) * 实际收益(t)  # 用前一天的信号，因为今天开盘前决定仓位
        """
        if signals is not None:
            self.signals = signals
        
        if self.signals is None:
            raise ValueError("请先调用 generate_signals() 或传入 signals")
        
        # 关键：信号滞后一天（今天收盘的信号，明天开盘才能交易）。
        # 与手续费计算共用 _executed_positions，缺失信号统一按空仓处理。
        positions = self._executed_positions(self.signals)
        self.strategy_returns = positions * self.returns
        return self.strategy_returns
    
    def calculate_performance(self) -> Dict:
        """
        计算回测绩效指标
        返回: 字典，包含年化收益、波动率、夏普、最大回撤等
        """
        if self.strategy_returns is None:
            raise ValueError("请先调用 calculate_strategy_returns()")
        
        # 固定资产池：先验证再求均值。跳过一个资产等于静默改变权重，
        # 因此这里必须 skipna=False；缺失数据在验证阶段就会报错。
        self._validate_fixed_universe(self.strategy_returns, label="strategy returns")
        portfolio_returns = self.strategy_returns.mean(axis=1, skipna=False)

        # 组合净值：从 1.0 开始按简单收益率复利。
        # 尚未建仓/无数据的期间按 0 收益处理，避免净值出现 NaN。
        cum_returns = (1.0 + portfolio_returns.fillna(0.0)).cumprod()
        total_return = cum_returns.iloc[-1] - 1
        n_periods = portfolio_returns.count()
        annual_return = (
            (1.0 + total_return) ** (252.0 / n_periods) - 1
            if n_periods > 0
            else np.nan
        )

        # 年化波动率
        periodic_std = portfolio_returns.std(ddof=0)
        annual_vol = periodic_std * np.sqrt(252)

        # 夏普比率（假设无风险利率为0）：
        # 周期均值 / 周期标准差（ddof=0）× sqrt(252)；零波动返回 NaN。
        # 不能用“复利年化收益 / 年化波动率”，那会把复利效应混入风险调整收益。
        sharpe = (
            portfolio_returns.mean() / periodic_std * np.sqrt(252)
            if periodic_std > 0
            else np.nan
        )

        # 最大回撤：高点包含初始净值 1.0，否则第一个负收益的回撤会被算成 0
        cummax = cum_returns.cummax().clip(lower=1.0)
        drawdowns = cum_returns / cummax - 1.0
        max_drawdown = drawdowns.min()

        # 胜率
        win_rate = (
            (portfolio_returns > 0).sum() / n_periods if n_periods > 0 else np.nan
        )
        
        # 盈亏比
        avg_win = portfolio_returns[portfolio_returns > 0].mean()
        avg_loss = portfolio_returns[portfolio_returns < 0].mean()
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        performance = {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_return': total_return,
            'cum_returns': cum_returns
        }
        
        return performance

    
    def ema_crossover_signal(self, fast: int = 5, slow: int = 20) -> pd.DataFrame:
        """均线交叉信号：快线在上做多，否则空仓"""
        ema_fast = self.prices.ewm(span = fast,adjust = False).mean()
        ema_slow = self.prices.ewm(span = slow,adjust = False).mean()
        conditions = [
            ema_fast>ema_slow,
            ema_fast<ema_slow,
        ]
        choices = [1,-1]
        signals = pd.DataFrame(
            np.select(conditions,choices,default = 0),
            index = self.prices.index,
            columns = self.prices.columns
        )
        return signals

    def apply_signal_delay(self, signals: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
        """应用信号延迟（避免未来函数）"""
        return signals.shift(lag)

    @staticmethod
    def _executed_positions(signals: pd.DataFrame) -> pd.DataFrame:
        """t 日实际持仓 = t-1 的信号（``shift(1)``）；缺失信号明确按空仓（0）处理。

        这是策略收益与手续费共用的唯一持仓口径，禁止两处各算一套。
        """
        return signals.shift(1).fillna(0.0)

    @classmethod
    def _executed_trades(cls, signals: pd.DataFrame) -> pd.DataFrame:
        """实际发生的仓位变化，与 ``_executed_positions`` 完全同源。

        - 第一笔建仓从 0 仓位计起，不是免费获得的；
        - 缺失信号视为空仓，因此随后的有效信号按“重新建仓”计费；
        - 最后一个信号要到样本外才会执行，因此不计手续费。
        """
        positions = cls._executed_positions(signals)
        return positions.diff().fillna(0.0)

    def calculate_turnover(self, signals: pd.DataFrame) -> pd.Series:
        """计算每日换手率（按实际执行的仓位变化）"""
        changes = self._executed_trades(signals).abs().sum(axis=1)
        return changes / signals.shape[1]

    def calculate_commission(self, signals: pd.DataFrame, rate: float = 0.0015) -> pd.DataFrame:
        """计算交易手续费（与实际执行的仓位变化对齐）"""
        trades = self._executed_trades(signals).abs()
        return trades * rate

    def calculate_net_returns(self, signals: Optional[pd.DataFrame] = None, 
                              commission_rate: float = 0.0015) -> pd.DataFrame:
        """计算扣除手续费后的净收益"""
        if signals is not None:
            self.signals = signals
    
        # 毛收益
        gross_returns = self.calculate_strategy_returns()
    
        # 手续费
        commission = self.calculate_commission(self.signals, commission_rate)
    
        # 净收益
        return gross_returns - commission

    def calculate_performance_net(self, commission_rate: float = 0.0015) -> Dict:
        """计算扣除手续费后的绩效"""
        net_returns = self.calculate_net_returns(commission_rate=commission_rate)
    
        # 临时替换 strategy_returns
        original_returns = self.strategy_returns
        self.strategy_returns = net_returns
        perf = self.calculate_performance()
        self.strategy_returns = original_returns  # 恢复
    
        return perf

    

    def calculate_ic(self, factor: pd.DataFrame, 
                     forward_returns: Optional[pd.DataFrame] = None,
                     min_obs: int = 10) -> pd.Series:
        """
            计算因子的 Rank IC 序列（Spearman 相关性）
    
            Parameters:
            factor: 因子值，行=日期，列=股票
            forward_returns: 与 factor 同索引的“下一期”收益率
                （第 t 行代表 t → t+1 的收益），
                默认用 self.returns.shift(-1)
            min_obs: 参与单日 IC 计算的最少资产数
    
            Returns:
            Series: 每日 IC 值；因子最后一天因缺少未来收益不会出现在结果中
        """
        if forward_returns is None:
            # self.returns 是简单收益率；Rank IC 对每个资产的单调变换不敏感
            forward_returns = self.returns.shift(-1)

        ic_values = {}

        for date in factor.index:
            # 第 t 行的 forward_returns 已经是 t → t+1 的收益，
            # 不能再按“下一个交易日的 forward_returns”取值，否则会滞后两期。
            if date not in forward_returns.index:
                continue

            f = factor.loc[date].dropna()
            if len(f) < min_obs:
                continue

            r = forward_returns.loc[date]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            r = r.dropna()

            # 取交集
            common = f.index.intersection(r.index)
            if len(common) >= min_obs:
                f_common = f[common]
                r_common = r[common]
                # 常数截面（例如整行 0 收益）没有秩相关，跳过以避免 NaN 警告
                if f_common.nunique() < 2 or r_common.nunique() < 2:
                    continue
                ic, _ = spearmanr(f_common, r_common)
                if not np.isnan(ic):
                    ic_values[date] = ic

        return pd.Series(ic_values, name='IC', dtype='float64')


    def calculate_ic_summary(self, factor: pd.DataFrame) -> Dict:
        """计算因子 IC 摘要统计"""
        ic_series = self.calculate_ic(factor)

        n = len(ic_series)
        ic_mean = ic_series.mean()
        ic_std = ic_series.std(ddof=0)  # 总体标准差（与波动率计算一致）

        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_mean / ic_std if ic_std > 0 else 0,  # Information Ratio
            'ic_win_rate': (ic_series > 0).mean(),           # IC>0 的比例
            'ic_t_stat': ic_mean / ic_std * np.sqrt(n) if ic_std > 0 else 0,  # t统计量
            'ic_positive_days': (ic_series > 0).sum(),
            'ic_negative_days': (ic_series < 0).sum(),
            'n_obs': n
        }


    def combine_factors_equal_weight(self, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        等权组合多个因子（横截面标准化后加总）
        
        Parameters:
        factors: 字典 {因子名: 因子值DataFrame}
    
        Returns:
        DataFrame: 组合因子值
        """
        # 初始化（用第一个因子的结构）
        first_factor = list(factors.values())[0]
        combined = pd.DataFrame(0.0, index=first_factor.index, columns=first_factor.columns)
    
        for name, factor in factors.items():
            # 横截面标准化：每天的所有股票中，均值=0，标准差=1
            # axis=1 表示对每行（每个日期）计算跨股票的统计量
            zscore = factor.sub(factor.mean(axis=1), axis=0).div(factor.std(axis=1), axis=0)
            combined = combined + zscore.fillna(0)  # 处理 NaN
    
        return combined / len(factors)




    def plot_equity_curve(self, benchmark: bool = True, figsize: tuple = (12, 6)):
        """绘制净值曲线（简单收益率复利）"""
        perf = self.calculate_performance()
        cum_returns = perf['cum_returns']
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(cum_returns.index, cum_returns.values, 'b-', linewidth=1.5, label='Strategy')
        
        if benchmark:
            # 等权买入持有（从 1.0 开始按简单收益率复利）
            bh = (1.0 + self.returns.mean(axis=1).fillna(0.0)).cumprod()
            ax.plot(bh.index, bh.values, 'gray', linewidth=1, alpha=0.7, label='Buy & Hold')
        
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_title('Strategy Equity Curve')
        ax.set_ylabel('Net Value')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    
    def plot_drawdown(self, figsize: tuple = (12, 4)):
        """绘制回撤图（高点包含初始净值 1.0）"""
        perf = self.calculate_performance()
        cum_returns = perf['cum_returns']
        cummax = cum_returns.cummax().clip(lower=1.0)
        drawdowns = (cum_returns / cummax - 1.0) * 100
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.fill_between(drawdowns.index, 0, drawdowns.values, color='red', alpha=0.5)
        ax.set_title('Strategy Drawdown')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    
    def plot_monthly_heatmap(self, figsize: tuple = (12, 5)):
        """绘制月度收益热力图（简单收益率复利）"""
        if self.strategy_returns is None:
            raise ValueError("请先计算策略收益")
        
        # 等权组合简单收益率
        portfolio_returns = self.strategy_returns.mean(axis=1)
        
        # 按月复利：(1+r) 连乘后减 1
        monthly_returns = (
            (1.0 + portfolio_returns.fillna(0.0)).resample('ME').prod() - 1.0
        )
        monthly_matrix = monthly_returns.groupby(
            [monthly_returns.index.year, monthly_returns.index.month]
        ).first().unstack()
        
        if monthly_matrix.empty:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(monthly_matrix.values, cmap='RdYlGn', aspect='auto',
                       vmin=-0.1, vmax=0.1)
        
        # 设置标签
        months = ['Jan','Feb','Mar','Apr','May','Jun',
                  'Jul','Aug','Sep','Oct','Nov','Dec']
        ax.set_xticks(range(len(monthly_matrix.columns)))
        ax.set_xticklabels(months[:len(monthly_matrix.columns)])
        ax.set_yticks(range(len(monthly_matrix.index)))
        ax.set_yticklabels(monthly_matrix.index)
        ax.set_title('Monthly Returns Heatmap')
        
        plt.colorbar(im, ax=ax, format=mtick.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        return fig
    
    
    def plot_full_report(self, title: str = 'Strategy Backtest Report',
                         figsize: tuple = (14, 12)):
        """生成完整回测报告（三合一）"""
        perf = self.calculate_performance()
        cum_returns = perf['cum_returns']
        portfolio_returns = self.strategy_returns.mean(axis=1)
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # 1. 净值曲线
        axes[0].plot(cum_returns.index, cum_returns.values, 'b-', linewidth=1.5)
        axes[0].fill_between(cum_returns.index, 1, cum_returns.values,
                              where=(cum_returns.values >= 1), color='green', alpha=0.3)
        axes[0].fill_between(cum_returns.index, 1, cum_returns.values,
                              where=(cum_returns.values < 1), color='red', alpha=0.3)
        axes[0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
        axes[0].set_title(f'{title} - Equity Curve')
        axes[0].set_ylabel('Net Value')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 回撤图（高点包含初始净值 1.0）
        cummax = cum_returns.cummax().clip(lower=1.0)
        drawdowns = (cum_returns / cummax - 1.0) * 100
        axes[1].fill_between(drawdowns.index, 0, drawdowns.values, color='red', alpha=0.5)
        axes[1].set_title('Drawdown (%)')
        axes[1].set_ylabel('Drawdown %')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 月度收益热力图（简单收益率复利）
        monthly_returns = (
            (1.0 + portfolio_returns.fillna(0.0)).resample('ME').prod() - 1.0
        )
        monthly_matrix = monthly_returns.groupby(
            [monthly_returns.index.year, monthly_returns.index.month]
        ).first().unstack()
        
        if not monthly_matrix.empty:
            im = axes[2].imshow(monthly_matrix.values, cmap='RdYlGn', aspect='auto',
                               vmin=-0.1, vmax=0.1)
            months = ['Jan','Feb','Mar','Apr','May','Jun',
                      'Jul','Aug','Sep','Oct','Nov','Dec']
            axes[2].set_xticks(range(len(monthly_matrix.columns)))
            axes[2].set_xticklabels(months[:len(monthly_matrix.columns)])
            axes[2].set_yticks(range(len(monthly_matrix.index)))
            axes[2].set_yticklabels(monthly_matrix.index)
            axes[2].set_title('Monthly Returns Heatmap')
            plt.colorbar(im, ax=axes[2], format=mtick.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        plt.tight_layout()
        return fig
    
    
    def parameter_sensitivity(self, param_name: str, param_range: list,
                              factor_func: Callable, threshold: float = 0,
                              **kwargs) -> pd.DataFrame:
        """
        参数敏感性分析（EWMA 友好版本）
        
        Parameters:
        param_name: 参数名称（如 'span', 'window'）
        param_range: 参数取值范围
        factor_func: 因子计算函数，接受参数返回因子 DataFrame
        threshold: 多空阈值（默认 0）
        """
        results = []
        
        for param_val in param_range:
            # 计算因子
            factor = factor_func(param_val, **kwargs)
            
            # 生成信号
            signals = self.generate_signals(factor, long_threshold=threshold)
            
            # 计算策略收益
            self.calculate_strategy_returns(signals)
            
            # 计算绩效
            perf = self.calculate_performance()
            
            results.append({
                param_name: param_val,
                'sharpe': perf['sharpe_ratio'],
                'annual_return': perf['annual_return'],
                'max_drawdown': perf['max_drawdown'],
                'win_rate': perf['win_rate']
            })
        
        return pd.DataFrame(results)
    
    
    # ============================================================
    # 使用 EWMA 的因子函数示例
    # ============================================================
    
    def momentum_factor_ewma(span: int, **kwargs) -> pd.DataFrame:
        """EWMA 动量因子：价格 / EWMA 均价 - 1"""
        prices = kwargs.get('prices')  # 需要传入 prices
        ema = prices.ewm(span=span, adjust=False).mean()
        return prices / ema - 1
    
    # 使用示例：
    # bt = FactorBacktest(prices)
    # results = bt.parameter_sensitivity(
    #     'span', range(5, 125, 5),
    #     lambda s: bt.prices / bt.prices.ewm(span=s, adjust=False).mean() - 1
    # )