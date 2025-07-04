import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig

# 프로젝트 루트 경로 설정
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

@hydra.main(version_base=None, config_path="../configs_hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    """간단한 Hydra Config 테스트"""
    
    print("=== Hydra Config 테스트 ===")
    print(f"실험명: {cfg.exp_name}")
    print(f"GPU: {cfg.gpu}")
    print(f"Workers: {cfg.workers}")
    print(f"Seed: {cfg.seed}")
    
    print("\n=== Dataset Config ===")
    print(cfg.dataset)
    
    print("\n=== Model Config ===") 
    print(cfg.model)
    
    print("\n=== Federated Config ===")
    print(cfg.federated)
    
    print("\n=== Training Config ===")
    print(cfg.training)
    
    print("\n=== 전체 Config ===")
    print(cfg)

if __name__ == "__main__":
    main() 