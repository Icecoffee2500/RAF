"""
Client Package - 연합학습 클라이언트

리팩토링된 FLClient는 기존의 533줄에서 200줄 미만으로 단순화되었으며,
각 책임을 전담하는 매니저 컴포넌트들을 조합하는 오케스트레이터 역할만 수행합니다.
"""

from .fl_client import FLClient

__all__ = ["FLClient"] 