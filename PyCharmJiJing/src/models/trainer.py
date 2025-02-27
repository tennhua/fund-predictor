import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple
import mlflow
import logging

class ModelTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
    def train(self, model: nn.Module, train_data: Tuple[np.ndarray, np.ndarray],
              valid_data: Tuple[np.ndarray, np.ndarray]) -> nn.Module:
        """
        训练模型
        """
        try:
            # 设置MLflow跟踪
            mlflow.start_run()
            
            # 准备数据加载器
            train_loader = self._prepare_dataloader(train_data)
            valid_loader = self._prepare_dataloader(valid_data)
            
            # 初始化优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            criterion = nn.MSELoss()
            
            # 训练循环
            for epoch in range(self.config['epochs']):
                train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
                valid_loss = self._validate(model, valid_loader, criterion)
                
                # 记录指标
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'valid_loss': valid_loss
                }, step=epoch)
                
                self.logger.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}')
                
            mlflow.end_run()
            return model
            
        except Exception as e:
            self.logger.error(f"训练错误: {str(e)}")
            raise
            
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader,
                     optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader) 