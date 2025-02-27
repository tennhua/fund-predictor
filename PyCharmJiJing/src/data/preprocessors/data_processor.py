import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
import logging

class DataPreprocessor:
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)
        
    def process_fund_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理基金数据
        """
        try:
            # 处理缺失值
            df = self._handle_missing_values(df)
            
            # 特征工程
            df = self._create_features(df)
            
            # 数据标准化
            features = self._normalize_features(df)
            
            # 创建时间序列数据
            X, y = self._create_sequences(features)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"数据预处理错误: {str(e)}")
            raise
            
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # 使用前向填充处理缺失值
        return df.fillna(method='ffill')
        
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 添加技术指标
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['RSI'] = self._calculate_rsi(df['close'])
        return df
        
    def _normalize_features(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(df)
        
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        sequence_length = self.config['sequence_length']
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length, 0])  # 预测收盘价
            
        return np.array(X), np.array(y) 