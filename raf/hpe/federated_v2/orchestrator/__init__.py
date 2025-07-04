"""
Orchestrator Package - 연합학습 전체 프로세스 관리

FLOrchestrator는 기존의 복잡한 main() 함수(308줄)를 대체하여
전체 연합학습 프로세스를 체계적으로 관리합니다.
"""

from .fl_orchestrator import FLOrchestrator

__all__ = ["FLOrchestrator"] 