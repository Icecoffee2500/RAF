"""
Core Components - 핵심 기능 컴포넌트들

각 매니저는 특정 책임을 가집니다:
- DataManager: 데이터 로딩 및 관리
- ModelManager: 모델 생성 및 관리  
- TrainingManager: 훈련 로직
- EvaluationManager: 평가 로직
- LoggingManager: 로깅 및 모니터링
"""

from .data_manager import DataManager
from .model_manager import ModelManager
from .training_manager import TrainingManager
from .evaluation_manager import EvaluationManager
from .logging_manager import LoggingManager

__all__ = [
    "DataManager",
    "ModelManager", 
    "TrainingManager",
    "EvaluationManager",
    "LoggingManager"
] 