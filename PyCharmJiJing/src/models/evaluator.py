import numpy as np
from typing import Dict, List
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class ModelEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        评估模型性能
        """
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            
            # 生成评估报告
            self._generate_evaluation_report(y_true, y_pred, metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"评估错误: {str(e)}")
            raise
            
    def _generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  metrics: Dict) -> None:
        # 创建可视化
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='实际值')
        plt.plot(y_pred, label='预测值')
        plt.title('预测结果对比')
        plt.legend()
        plt.savefig(f"{self.config['output_dir']}/prediction_comparison.png") 