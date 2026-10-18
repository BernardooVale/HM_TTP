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
    """Factory/Registry para solucionadores com suporte a resolução multi-processo."""
    _registry: dict[str, type[BaseSolver]] = {}
    _registry_modules: dict[str, tuple[str, str]] = {}
    
    @classmethod
    def register(cls, solver_class: type[BaseSolver]) -> type[BaseSolver]:
        """Decorator to register a solver."""
        instance = solver_class()
        cls._registry[instance.nome] = solver_class
        cls._registry_modules[instance.nome] = (solver_class.__module__, solver_class.__name__)
        return solver_class
    
    @classmethod
    def get(cls, nome: str) -> BaseSolver:
        if nome not in cls._registry:
            if nome in cls._registry_modules:
                mod_name, class_name = cls._registry_modules[nome]
                import importlib
                mod = importlib.import_module(mod_name)
                solver_cls = getattr(mod, class_name)
                cls._registry[nome] = solver_cls
            else:
                raise ValueError(f"Solucionador '{nome}' não encontrado. Disponíveis: {list(cls._registry.keys())}")
        return cls._registry[nome]()
    
    @classmethod
    def listar(cls) -> list[str]:
        return list(cls._registry.keys())
    
    @classmethod
    def todos(cls) -> list[BaseSolver]:
        return [cls._registry[nome]() for nome in cls._registry]
