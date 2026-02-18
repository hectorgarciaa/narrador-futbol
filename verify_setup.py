#!/usr/bin/env python
"""
Script de verificación del setup del proyecto.
Comprueba dependencias, módulos, configuración y assets.
"""
import sys
import os
from pathlib import Path
from importlib import import_module
from typing import Dict, List, Tuple

# Colores para terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text: str):
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")

# ============================================================================
# CHECKS
# ============================================================================

def check_python_version() -> bool:
    """Check Python version >= 3.8."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    if sys.version_info >= (3, 8):
        print_success(f"Python {version} (>=3.8 requerido)")
        return True
    else:
        print_error(f"Python {version} — Se requiere Python >=3.8")
        return False

def check_venv() -> bool:
    """Check if running inside a virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        venv_path = sys.prefix
        print_success(f"Entorno virtual activo: {venv_path}")
        return True
    else:
        print_warning("No se detecta venv activo — se recomienda activar con: .venv\\Scripts\\Activate.ps1")
        return False

def check_dependencies() -> Tuple[bool, Dict[str, str]]:
    """Check core dependencies."""
    required = {
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'supervision': 'supervision',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'yaml': 'pyyaml',
        'dotenv': 'python-dotenv',
    }
    
    results = {}
    all_ok = True
    
    for module_name, package_name in required.items():
        try:
            mod = import_module(module_name)
            version = getattr(mod, '__version__', 'unknown')
            results[package_name] = version
            print_success(f"{package_name:25} {version}")
        except ImportError:
            results[package_name] = "NOT INSTALLED"
            print_error(f"{package_name:25} NOT INSTALLED")
            all_ok = False
    
    return all_ok, results

def check_football_ai_imports() -> bool:
    """Check football_ai package imports."""
    imports = [
        'football_ai.core.config',
        'football_ai.detection.detector',
        'football_ai.tracking.tracker',
        'football_ai.identification.team_detector',
        'football_ai.visualization.drawer',
        'football_ai.evaluation.evaluator',
    ]
    
    all_ok = True
    for module_path in imports:
        try:
            import_module(module_path)
            print_success(f"{module_path}")
        except Exception as e:
            print_error(f"{module_path}: {str(e)[:60]}")
            all_ok = False
    
    return all_ok

def check_torch_cuda() -> Dict:
    """Check PyTorch CUDA support."""
    import torch
    
    results = {
        'version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    print_info(f"Versión: {results['version']}")
    
    if results['cuda_available']:
        print_success(f"CUDA disponible (versión {results['cuda_version']}, {results['device_count']} device(s))")
    else:
        print_warning("CUDA no disponible — ejecutando en CPU (más lento para ML)")
    
    return results

def check_config_files() -> bool:
    """Check existence of config files."""
    project_root = Path(__file__).parent
    files_to_check = {
        'config.yaml': project_root / 'config.yaml',
        '.env': project_root / '.env',
        'requirements.txt': project_root / 'requirements.txt',
        'pyproject.toml': project_root / 'pyproject.toml',
    }
    
    all_ok = True
    for name, path in files_to_check.items():
        if path.exists():
            print_success(f"{name}")
        else:
            print_error(f"{name} — NO ENCONTRADO")
            all_ok = False
    
    return all_ok

def check_env_vars() -> bool:
    """Check .env file has required variables."""
    dotenv_file = Path(__file__).parent / '.env'
    
    if not dotenv_file.exists():
        print_error(".env no existe")
        return False
    
    required_keys = [
        'ROBOFLOW_API_KEY',
        'ROBOFLOW_WORKSPACE',
        'ROBOFLOW_PROJECT',
    ]
    
    with open(dotenv_file) as f:
        content = f.read()
    
    all_ok = True
    for key in required_keys:
        if key in content:
            value = [line.split('=')[1].strip() for line in content.split('\n') if line.startswith(key)]
            if value and value[0] != 'your_api_key_here':
                print_success(f"{key}: configurado")
            else:
                print_warning(f"{key}: aún es placeholder (edita .env)")
                all_ok = False
        else:
            print_error(f"{key}: no encontrado en .env")
            all_ok = False
    
    return all_ok

def check_model_paths() -> bool:
    """Check if at least config references valid model paths."""
    try:
        from football_ai.core.config import get_config
        config = get_config()
        
        # Just check that config loads, actual model files are large and can be downloaded
        print_success("config.yaml carga correctamente")
        
        print_info("Modelos (verifica): yolo/v11/, yolo/v8/, finetuning/, finetuning-balon/")
        print_info("Usa scripts/data/download_models.py para descargar modelos")
        
        return True
    except Exception as e:
        print_error(f"Error al cargar config: {str(e)[:60]}")
        return False

def check_data_structure() -> bool:
    """Check data directory structure."""
    project_root = Path(__file__).parent
    dirs_to_check = {
        'data/': project_root / 'data',
        'data/detection/': project_root / 'data' / 'detection',
        'data/partidoPrueba/': project_root / 'data' / 'partidoPrueba',
        'scripts/': project_root / 'scripts',
        'football_ai/': project_root / 'football_ai',
        'models/': project_root / 'models',
    }
    
    all_ok = True
    for name, path in dirs_to_check.items():
        if path.exists():
            print_success(f"{name}")
        else:
            print_warning(f"{name} — no existe (se creará según sea necesario)")
    
    return all_ok

# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("🔍 VERIFICACIÓN DEL PROYECTO NARRADOR-FUTBOL")
    
    checks: List[Tuple[str, callable]] = [
        ("Python", check_python_version),
        ("Entorno Virtual", check_venv),
        ("Configuración y Archivos", check_config_files),
        ("Estructura de Directorios", check_data_structure),
        ("Dependencias Principales", lambda: check_dependencies()[0]),
        ("PyTorch y CUDA", check_torch_cuda),
        ("Módulos football_ai", check_football_ai_imports),
        ("Variables de Entorno (.env)", check_env_vars),
        ("Configuración (config.yaml)", check_model_paths),
    ]
    
    results: Dict[str, bool] = {}
    
    for check_name, check_func in checks:
        print_header(f"→ {check_name}")
        try:
            result = check_func()
            results[check_name] = result if isinstance(result, bool) else True
        except Exception as e:
            print_error(f"Error al ejecutar verificación: {str(e)[:80]}")
            results[check_name] = False
    
    # Summary
    print_header("📋 RESUMEN")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, result in results.items():
        symbol = Colors.GREEN + "✓" + Colors.RESET if result else Colors.RED + "✗" + Colors.RESET
        print(f"{symbol} {check_name}")
    
    print(f"\n{Colors.BOLD}Pasos completados: {passed}/{total}{Colors.RESET}\n")
    
    if all(results.values()):
        print_success("Todos los chequeos pasaron. ¡Listo para correr!")
        print_info("Próximos pasos:")
        print_info("1. Edita .env con tu API key de Roboflow")
        print_info("2. Descarga modelos: python scripts/data/download_models.py")
        print_info("3. Descarga datasets: python scripts/data/download_datasets.py")
        print_info("4. Ejecuta: python scripts/track.py")
        return 0
    else:
        print_error("Algunos chequeos fallaron. Ver errores arriba.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
