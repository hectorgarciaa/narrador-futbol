"""
Script de limpieza del proyecto narrador-futbol.

Elimina todos los archivos generados y descargables que se pueden regenerar:
- Caché de Python (__pycache__, *.pyc)
- Archivos de output (videos procesados, JSONs de tracking)
- Datasets descargables
- Modelos YOLO base (conserva modelos fine-tuned)
- Archivos de log

USO:
    python clean_project.py [--all] [--cache] [--output] [--datasets] [--models]
    
OPCIONES:
    --all       : Limpia todo (equivalente a todas las opciones)
    --cache     : Solo elimina caché de Python
    --output    : Solo elimina archivos de output/
    --datasets  : Solo elimina datasets de data/detection/
    --models    : Solo elimina modelos YOLO base
    --logs      : Solo elimina archivos .log
    
EJEMPLOS:
    python clean_project.py --cache --output   # Limpia caché y output
    python clean_project.py --all              # Limpieza completa
"""

import os
import shutil
import argparse
from pathlib import Path


def get_project_root():
    """Obtiene la raíz del proyecto."""
    return Path(__file__).parent


def get_dir_size(path):
    """Calcula el tamaño de un directorio en MB."""
    total = 0
    try:
        for entry in Path(path).rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception:
        pass
    return total / (1024 * 1024)  # Convertir a MB


def clean_cache(project_root):
    """Elimina caché de Python."""
    print("\n🧹 Limpiando caché de Python...")
    
    cache_dirs = list(project_root.rglob('__pycache__'))
    pyc_files = list(project_root.rglob('*.pyc'))
    pyo_files = list(project_root.rglob('*.pyo'))
    
    count = 0
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            count += 1
            print(f"  ✓ Eliminado: {cache_dir.relative_to(project_root)}")
        except Exception as e:
            print(f"  ✗ Error eliminando {cache_dir}: {e}")
    
    for pyc_file in pyc_files + pyo_files:
        try:
            pyc_file.unlink()
            count += 1
        except Exception:
            pass
    
    print(f"✅ Eliminados {count} archivos/directorios de caché")


def clean_output(project_root):
    """Elimina archivos de output generados."""
    print("\n🧹 Limpiando archivos de output...")
    
    output_dir = project_root / "output"
    if not output_dir.exists():
        print("  ℹ️  Directorio output/ no existe")
        return
    
    size_before = get_dir_size(output_dir)
    
    try:
        # Eliminar contenido pero mantener estructura
        for item in output_dir.rglob('*'):
            if item.is_file():
                item.unlink()
                print(f"  ✓ Eliminado: {item.relative_to(project_root)}")
        
        # Eliminar directorios vacíos excepto output/
        for item in sorted(output_dir.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if item.is_dir() and not any(item.iterdir()):
                item.rmdir()
        
        print(f"✅ Liberados ~{size_before:.1f} MB de output/")
        
        # Crear .gitkeep para mantener estructura
        (output_dir / ".gitkeep").touch()
        
    except Exception as e:
        print(f"  ✗ Error limpiando output/: {e}")


def clean_datasets(project_root):
    """Elimina datasets descargables."""
    print("\n🧹 Limpiando datasets...")
    
    detection_dir = project_root / "data" / "detection"
    if not detection_dir.exists():
        print("  ℹ️  Directorio data/detection/ no existe")
        return
    
    size_before = get_dir_size(detection_dir)
    
    try:
        for item in detection_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                print(f"  ✓ Eliminado dataset: {item.name}")
        
        print(f"✅ Liberados ~{size_before:.1f} MB de datasets")
        print("  ℹ️  Para regenerar: python scripts/data/download_datasets.py")
        
        # Crear .gitkeep
        (detection_dir / ".gitkeep").touch()
        
    except Exception as e:
        print(f"  ✗ Error limpiando datasets: {e}")


def clean_yolo_models(project_root):
    """Elimina modelos YOLO base (conserva fine-tuned)."""
    print("\n🧹 Limpiando modelos YOLO base...")
    
    yolo_dir = project_root / "models" / "yolo"
    if not yolo_dir.exists():
        print("  ℹ️  Directorio models/yolo/ no existe")
        return
    
    size_before = get_dir_size(yolo_dir)
    
    try:
        for item in yolo_dir.rglob('*.pt'):
            item.unlink()
            print(f"  ✓ Eliminado: {item.relative_to(project_root)}")
        
        print(f"✅ Liberados ~{size_before:.1f} MB de modelos YOLO")
        print("  ℹ️  Para regenerar: python scripts/data/download_models.py")
        
    except Exception as e:
        print(f"  ✗ Error limpiando modelos YOLO: {e}")


def clean_egg_info(project_root):
    """Elimina archivos egg-info generados por pip install -e ."""
    print("\n🧹 Limpiando egg-info...")
    
    egg_info_dirs = list(project_root.glob('*.egg-info'))
    
    count = 0
    for egg_dir in egg_info_dirs:
        try:
            shutil.rmtree(egg_dir)
            count += 1
            print(f"  ✓ Eliminado: {egg_dir.name}")
        except Exception as e:
            print(f"  ✗ Error eliminando {egg_dir}: {e}")
    
    if count > 0:
        print(f"✅ Eliminados {count} directorios egg-info")
    else:
        print("  ℹ️  No hay directorios egg-info para eliminar")


def clean_logs(project_root):
    """Elimina archivos de log."""
    print("\n🧹 Limpiando archivos de log...")
    
    log_files = list(project_root.rglob('*.log'))
    
    count = 0
    for log_file in log_files:
        try:
            log_file.unlink()
            count += 1
            print(f"  ✓ Eliminado: {log_file.relative_to(project_root)}")
        except Exception as e:
            print(f"  ✗ Error eliminando {log_file}: {e}")
    
    if count > 0:
        print(f"✅ Eliminados {count} archivos de log")
    else:
        print("  ℹ️  No hay archivos de log para eliminar")


def main():
    parser = argparse.ArgumentParser(
        description="Limpia archivos generados del proyecto narrador-futbol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Limpia todo (caché, output, datasets, modelos)')
    parser.add_argument('--cache', action='store_true',
                       help='Limpia caché de Python')
    parser.add_argument('--output', action='store_true',
                       help='Limpia archivos de output/')
    parser.add_argument('--datasets', action='store_true',
                       help='Limpia datasets descargables')
    parser.add_argument('--models', action='store_true',
                       help='Limpia modelos YOLO base')
    parser.add_argument('--logs', action='store_true',
                       help='Limpia archivos de log')
    parser.add_argument('--egg-info', action='store_true',
                       help='Limpia directorios egg-info generados por pip')
    
    args = parser.parse_args()
    
    # Si no se especifica nada, mostrar ayuda
    if not any([args.all, args.cache, args.output, args.datasets, args.models, args.logs, getattr(args, 'egg_info', False)]):
        parser.print_help()
        return
    
    project_root = get_project_root()
    print(f"\n📁 Proyecto: {project_root}")
    print("="*60)
    
    if args.all:
        args.cache = args.output = args.datasets = args.models = args.logs = True
        args.egg_info = True
    
    if args.cache:
        clean_cache(project_root)
    
    if args.output:
        clean_output(project_root)
    
    if args.datasets:
        clean_datasets(project_root)
    
    if args.models:
        clean_yolo_models(project_root)
    
    if args.logs:
        clean_logs(project_root)
    
    if getattr(args, 'egg_info', False):
        clean_egg_info(project_root)
    
    print("\n" + "="*60)
    print("✨ Limpieza completada")
    print("\n💡 Tip: Ejecuta 'python clean_project.py --help' para ver todas las opciones")


if __name__ == "__main__":
    main()
