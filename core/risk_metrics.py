""" risk metrics  module
date: 2026-04-20
description :rolling_risk metrics for financial time series
""" 

import pandas as pd
import numpy as np
from typing import Optional, Union

class RiskMetrics:
    def __init__(self,returns:pd.DataFrame):
        self.returns = returns
        self.prices = (1+returns).cumprod()*100
        #(这里是用来计算离散的收益率和价格，如果是连续的话，应该写成 np.exp(np.cumsum(returns))*100

    def rolling_volatility(self,window:int = 20,annualize : bool = True) -> pd.DataFrame:

        vol = self.returns.rolling(window,min_periods = window//2).std(ddof = 0)
        #minperiods = window/2,的意思是，最少要有十个数字，才会进行计算，否则返回nan
        if annualize:
            vol = vol*np.sqrt(252)
        return vol
    def rolling_sharpe(self,window:int = 20,annualize : bool = True,rf_rate : float = 0.0) -> pd.DataFrame:
        rf_daily = rf_rate/252

        excess_returns = self.returns - rf_daily
        mean_excess = excess_returns.rolling(window,min_periods = window//2).mean()
        vol_excess = excess_returns.rolling(window,min_periods = window//2).std(ddof = 0)

        sharpe = mean_excess / vol_excess
        # 零波动没有定义的 Sharpe（不是 ±inf），返回 NaN
        sharpe = sharpe.mask(vol_excess == 0)
        if annualize:
            # 周期 Sharpe × sqrt(252)；滚动均值/标准差本身不做年化
            sharpe = sharpe * np.sqrt(252)
        return sharpe

    def rolling_max_drawdown(self,window:int = 20) ->pd.DataFrame:
        """每个尾随收益窗口内的最大回撤。

        每个窗口自成一个相对净值序列：窗口开始时相对财富为 1.0，只用窗口
        内的收益复利，再对窗口内运行高点计算回撤。这样 [-20%, 0] 之后的
        窗口若收益为 0，回撤会回到 0%，而不是把窗口外的历史高点带进来。
        窗口开头的补齐 NaN 允许存在；中间缺口返回 NaN。
        """
        def calc_mdd(window_returns: np.ndarray) -> float:
            mask = ~np.isnan(window_returns)
            if not mask.any():
                return np.nan
            first = int(np.argmax(mask))
            if not mask[first:].all():
                return np.nan
            valid = window_returns[first:]
            wealth = np.concatenate(([1.0], np.cumprod(1.0 + valid)))
            peak = np.maximum.accumulate(wealth)
            return float(np.min(wealth / peak - 1.0))

        return self.returns.rolling(window,min_periods = window//2).apply(calc_mdd, raw=True)

    def historical_var(self,confidence:float = 0.95,window:Optional[int] = None) -> pd.Series:
        alpha = 1 - confidence
        if window is not None:
            return self.returns.rolling(window).quantile(alpha)
        else:
            return self.returns.quantile(alpha)

    def cumulative_returns(self) -> pd.DataFrame:
        
        """计算累积收益率"""
        return (1 + self.returns).cumprod()

    def expanding_volatility(self, annualize: bool = True) -> pd.DataFrame:
        """计算累积波动率（从起始到当前）"""
        vol = self.returns.expanding().std(ddof=0)
        if annualize:
            vol = vol * np.sqrt(252)
        return vol

    def current_drawdown(self) -> pd.DataFrame:
        """计算当前回撤（高点包含初始净值 100）"""
        cummax = self.prices.cummax().clip(lower=100.0)
        return (self.prices - cummax) / cummax

    def expanding_sharpe(self, rf_rate: float = 0.0) -> pd.DataFrame:
        """计算累积夏普比率（零波动返回 NaN）"""
        rf_daily = rf_rate / 252
        excess = self.returns - rf_daily
    
        cum_mean = excess.expanding().mean()
        cum_std = excess.expanding().std(ddof=0)

        sharpe = (cum_mean / cum_std).mask(cum_std == 0)
        return sharpe * np.sqrt(252)

    # 在 RiskMetrics 类中添加

    def ewma_volatility(self, lambda_param: float = 0.94, annualize: bool = True) -> pd.DataFrame:
        """
            计算 EWMA 波动率（RiskMetrics 模型）
            lambda_param: 衰减因子，日频通常 0.94，月频 0.97
        """
        span = (2 / (1 - lambda_param)) - 1
    
        # EWMA 方差
        squared_returns = self.returns ** 2
        ewma_var = squared_returns.ewm(span=span, adjust=False).mean()
    
        # 波动率
        vol = np.sqrt(ewma_var)
    
        if annualize:
            vol = vol * np.sqrt(252)
    
        return vol

    def ewma_sharpe(self,lambda_param:float = 0.94,annualize:bool = True,rf_rate:float = 0.0) -> pd.DataFrame:

        """EWMA Sharpe ratio.

        使用指数加权方差 Var_t[r] = E_t[r^2] - (E_t[r])^2，而不是均方根
        sqrt(E_t[r^2])（后者把未中心化的二阶矩当成了方差）。
        """
        span = (2 / (1 - lambda_param)) - 1
        rf_daily = rf_rate/252
        excess = self.returns - rf_daily

        ewma_sharpe_mean = excess.ewm(span = span,adjust = False).mean()
        ewma_sharpe_var = (excess**2).ewm(span = span,adjust = False).mean() - ewma_sharpe_mean**2
        ewma_sharpe_vol = np.sqrt(ewma_sharpe_var.clip(lower=0))

        sharpe = ewma_sharpe_mean/ewma_sharpe_vol
        # 零波动没有定义的 Sharpe（不是 ±inf），返回 NaN
        sharpe = sharpe.mask(ewma_sharpe_vol == 0)
        if annualize:
            sharpe = sharpe*np.sqrt(252)
        return sharpe

    def ewma_covariance(self, lambda_param: float = 0.94) -> pd.DataFrame:
        """
            计算 EWMA 协方差矩阵（MultiIndex）
            返回：DataFrame with MultiIndex (date, asset)
        """
        span = (2 / (1 - lambda_param)) - 1
        return self.returns.ewm(span=span, adjust=False).cov()

    def rolling_beta(self, asset: str, benchmark: str, window: int = 60) -> pd.Series:
        """
            计算滚动 Beta（资产相对于基准的敏感度）
            Beta = Cov(asset, benchmark) / Var(benchmark)

            协方差与方差使用相同的自由度（ddof=0），否则
            Beta 会被放大 n/(n-1) 倍。
        """
        # 滚动协方差
        cov = self.returns[asset].rolling(window).cov(self.returns[benchmark], ddof=0)
        # 滚动方差
        var = self.returns[benchmark].rolling(window).var(ddof=0)
    
        return cov / var

    def ewma_beta(self, asset: str, benchmark: str, lambda_param: float = 0.94) -> pd.Series:
        """
            计算 EWMA Beta
        """
        span = (2 / (1 - lambda_param)) - 1
    
        # EWMA 协方差
        cov = self.returns[asset].ewm(span=span, adjust=False).cov(self.returns[benchmark])
        # EWMA 方差
        var = self.returns[benchmark].ewm(span=span, adjust=False).var()
    
        return cov / var
        

class RiskReport:
    """
    风险报告生成器
    """
    def __init__(self, risk_metrics: RiskMetrics):
        self.rm = risk_metrics
        self.returns = risk_metrics.returns
        
    def generate_summary(self, date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        生成某一天的风险摘要表
        date: 目标日期，None 表示最新日期
        """
        if date is None:
            date = self.returns.index[-1]
        
        # 计算各项指标
        vol_20 = self.rm.rolling_volatility(20).loc[date]
        sharpe_20 = self.rm.rolling_sharpe(20).loc[date]
        dd = self.rm.current_drawdown().loc[date]
        ewma_vol = self.rm.ewma_volatility().loc[date]
        
        # 高波动判断
        hist_vol = self.rm.ewma_volatility()
        threshold = hist_vol.expanding().quantile(0.95).loc[date]
        is_high_vol = ewma_vol > threshold
        
        # VaR 突破（过去5天）
        var_95 = self.rm.historical_var(confidence=0.95, window=252)
        recent_breach = (self.returns.loc[:date].tail(5) < var_95.loc[:date].tail(5)).sum()
        
        # 汇总
        summary = pd.DataFrame({
            'Volatility_20d_Ann': vol_20,
            'Sharpe_20d': sharpe_20,
            'Current_Drawdown': dd,
            'EWMA_Vol_Ann': ewma_vol,
            'High_Vol_Flag': is_high_vol,
            'VaR_Breach_5d': recent_breach
        })
        
        return summary.round(4)
    
    def generate_html_report(self, output_path: str = 'risk_report.html'):
        """生成 HTML 格式的风险报告"""
        summary = self.generate_summary()
        
        # 转 HTML
        html = f"""
        <html>
        <head><title>Risk Report - {pd.Timestamp.now().strftime('%Y-%m-%d')}</title></head>
        <body>
            <h1>Daily Risk Summary</h1>
            <h2>As of {self.returns.index[-1].strftime('%Y-%m-%d')}</h2>
            {summary.to_html()}
            
            <h2>Volatility Chart (Last 60 Days)</h2>
            <img src="vol_chart.png" width="100%">
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"✅ 报告已生成: {output_path}")