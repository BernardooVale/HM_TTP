from abc import ABC, abstractmethod
from src.core.models import TTPInstance, SolucaoTTP

class BaseSolver(ABC):
    """Interface abstrata para solucionadores do TTP."""
    
    @property
    @abstractmethod
    def nome(self) -> str:
        """Nome identificador do solucionador."""
        ...
    
    @abstractmethod
    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        """Resolve uma instância do TTP e retorna a solução."""
        ...

class SolverRegistry:
    """Factory/Registry para solucionadores."""
    _registry: dict[str, type[BaseSolver]] = {}
    
    @classmethod
    def register(cls, solver_class: type[BaseSolver]) -> type[BaseSolver]:
        """Decorator to register a solver."""
        instance = solver_class()
        cls._registry[instance.nome] = solver_class
        return solver_class
    
    @classmethod
    def get(cls, nome: str) -> BaseSolver:
        if nome not in cls._registry:
            raise ValueError(f"Solucionador '{nome}' não encontrado. Disponíveis: {list(cls._registry.keys())}")
        return cls._registry[nome]()
    
    @classmethod
    def listar(cls) -> list[str]:
        return list(cls._registry.keys())
    
    @classmethod
    def todos(cls) -> list[BaseSolver]:
        return [cls._registry[nome]() for nome in cls._registry]
