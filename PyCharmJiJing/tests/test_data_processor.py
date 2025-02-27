import unittest
import pandas as pd
import numpy as np
from src.data.preprocessors.data_processor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.config = {
            'sequence_length': 10
        }
        self.preprocessor = DataPreprocessor(self.config)
        
    def test_handle_missing_values(self):
        # 创建测试数据
        test_data = pd.DataFrame({
            'close': [1.0, np.nan, 3.0, 4.0]
        })
        
        processed_data = self.preprocessor._handle_missing_values(test_data)
        self.assertFalse(processed_data.isnull().any().any())
        
    def test_create_sequences(self):
        test_data = np.array([[1], [2], [3], [4], [5]])
        X, y = self.preprocessor._create_sequences(test_data)
        
        self.assertEqual(X.shape[1], self.config['sequence_length']) 