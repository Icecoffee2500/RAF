"""
Federated Learning V2 - 리팩토링된 컴포넌트 기반 아키텍처

주요 특징:
- 책임 분리: 각 컴포넌트가 단일 책임을 가짐
- 의존성 주입: Hydra config가 main()에서만 주입되고 명시적으로 전달됨
- 테스트 가능성: 각 컴포넌트가 독립적으로 테스트 가능
- 확장성: 새로운 기능 추가가 용이
"""

from .client.fl_client import FLClient
from .orchestrator.fl_orchestrator import FLOrchestrator

__version__ = "2.0.0"
__all__ = ["FLClient", "FLOrchestrator"] 