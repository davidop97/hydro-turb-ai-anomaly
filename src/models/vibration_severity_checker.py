from typing import Dict, Optional

# Umbrales por defecto
DEFAULT_ACTION_LIMITS = {
    "Francis horizontal": {
        "GE-DE": {"verde": 100, "amarillo": 150, "rojo": 150},
        "GE-NDE": {"verde": 95, "amarillo": 150, "rojo": 150},
        "T": {"verde": 95, "amarillo": 150, "rojo": 150},
    },
    "Pelton horizontal": {
        "GE-DE": {"verde": 145, "amarillo": 225, "rojo": 225},
        "GE-NDE": {"verde": 95, "amarillo": 150, "rojo": 150},
        "T": {"verde": 95, "amarillo": 150, "rojo": 150},
    },
    "Pump horizontal": {
        "GE-DE": {"verde": 110, "amarillo": 170, "rojo": 170},
        "GE-NDE": {"verde": 110, "amarillo": 170, "rojo": 170},
        "T": {"verde": 95, "amarillo": 150, "rojo": 150},
    },
}

def check_vibration_severity(
    max_values: Dict[str, float],
    machine_type: str,
    severity_level: Optional[str] = None,
    custom_thresholds: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, str]:
    """
    Compara los valores máximos de vibración con los límites de acción y determina la severidad.
    """
    if machine_type not in DEFAULT_ACTION_LIMITS:
        raise ValueError(
            f"Tipo de máquina '{machine_type}' no reconocido. "
            f"Opciones: {list(DEFAULT_ACTION_LIMITS.keys())}"
        )

    if severity_level not in [None, 'verde', 'amarillo', 'rojo']:
        raise ValueError("severity_level debe ser 'verde', 'amarillo', 'rojo' o None.")

    # Preparar umbrales: combinar personalizados con predeterminados
    thresholds: Dict[str, Dict[str, float]] = {"GE-DE": {}, "GE-NDE": {}, "T": {}}
    for direction in ["GE-DE", "GE-NDE", "T"]:
        # Iniciar con los valores predeterminados
        thresholds[direction] = {
            key: float(value)
            for key, value in DEFAULT_ACTION_LIMITS[machine_type][direction].items()
        }

        # Si hay umbrales personalizados, aplicarlos
        if custom_thresholds and direction in custom_thresholds:
            if severity_level is None:
                # Modo general: se esperan umbrales para todos los colores
                for color in ["verde", "amarillo", "rojo"]:
                    if color in custom_thresholds[direction]:
                        thresholds[direction][color] = custom_thresholds[direction][color]
            else:
                # Modo específico: solo se ajusta el umbral del color seleccionado
                if severity_level in custom_thresholds[direction]:
                    thresholds[direction][severity_level] = (
                        custom_thresholds[direction][severity_level]
                    )

    # Mapeo por defecto:
    # CLE*, CS*, C1* -> GE-NDE; CLA*, CI*, C2*, CG* -> GE-DE; CT* -> T; otros -> desconocido
    mapping = {}
    for col in max_values.keys():
        if col.startswith(('CLE', 'CS', 'C1')):
            mapping[col] = 'GE-NDE'
        elif col.startswith(('CLA', 'CI', 'C2','CG')):
            mapping[col] = 'GE-DE'
        elif col.startswith('CT'):
            mapping[col] = 'T'
        else:
            mapping[col] = 'T'  # Asignar por defecto a GE-NDE si no se encuentra en el mapeo

    severity_results = {}
    for col, value in max_values.items():
        direction = mapping.get(col, 'unknown')
        if direction == 'unknown':
            severity_results[col] = 'desconocido'  # Asignar severidad predeterminada
            continue

        limit_verde = thresholds[direction]['verde']
        limit_amarillo = thresholds[direction]['amarillo']
        limit_rojo = thresholds[direction]['rojo']

        if severity_level == 'verde':
            severity_results[col] = 'verde' if value <= limit_verde else 'no cumple'
        elif severity_level == 'amarillo':
            severity_results[col] = (
                'amarillo' if limit_verde < value <= limit_amarillo else 'no cumple'
            )
        elif severity_level == 'rojo':
            severity_results[col] = 'rojo' if value > limit_rojo else 'no cumple'
        else:  # Clasificación general
            if value <= limit_verde:
                severity_results[col] = 'verde'
            elif limit_verde < value <= limit_amarillo:
                severity_results[col] = 'amarillo'
            else:
                severity_results[col] = 'rojo'

    return severity_results

# Ejemplo de implementación con ingreso manual de umbrales
if __name__ == "__main__":
    # Datos de ejemplo
    max_values = {
        'CLEX': 35.09,
        'CSL': 33.84,
        'CLAX': 29.73,
        'CIP': 25.89,
        'CTP': 50.0
    }
    machine_type = "Francis horizontal"

    # 1. Modo general: ajustar todos los umbrales
    print("=== Modo General (severity_level=None) ===")
    custom_thresholds_all = {
        "GE-DE": {"verde": 90.0, "amarillo": 140.0, "rojo": 140.0},
        "GE-NDE": {"verde": 80.0, "amarillo": 130.0, "rojo": 130.0},
        "T": {"verde": 80.0, "amarillo": 130.0, "rojo": 130.0}
    }
    result_all = check_vibration_severity(max_values, machine_type, None, custom_thresholds_all)
    print("Evaluación con todos los umbrales ajustados:", result_all)
    # Salida esperada: {'CLEX': 'verde', 'CSL': 'verde', 'CLAX': 'verde',
    #                    'CIP': 'verde', 'CTP': 'verde'}

    # 2. Modo específico: ajustar solo el umbral de 'verde'
    print("\n=== Modo Específico (severity_level='verde') ===")
    custom_thresholds_verde = {
        "GE-DE": {"verde": 30.0},  # Solo ajustamos 'verde', los otros permanecen predeterminados
        "GE-NDE": {"verde": 25.0},
        "T": {"verde": 40.0}
    }
    result_verde = check_vibration_severity(
        max_values, machine_type, 'verde', custom_thresholds_verde
    )
    print("Evaluación solo ajustando 'verde':", result_verde)
    # Salida esperada: 
    # {'CLEX': 'no cumple', 'CSL': 'no cumple', 
    # 'CLAX': 'no cumple', 'CIP': 'no cumple', 'CTP': 'no cumple'}

    # 3. Modo específico: ajustar solo el umbral de 'amarillo'
    print("\n=== Modo Específico (severity_level='amarillo') ===")
    custom_thresholds_amarillo = {
        "GE-DE": {"amarillo": 120.0},  # Solo ajustamos 'amarillo'
        "GE-NDE": {"amarillo": 110.0},
        "T": {"amarillo": 110.0}
    }
    result_amarillo = check_vibration_severity(
        max_values, 
        machine_type, 
        'amarillo', 
        custom_thresholds_amarillo
        )
    print("Evaluación solo ajustando 'amarillo':", result_amarillo)
    # Salida esperada: 
    # {'CLEX': 'no cumple', 'CSL': 'no cumple', 
    # 'CLAX': 'no cumple', 'CIP': 'no cumple', 'CTP': 'no cumple'}