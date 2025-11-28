"""
Workflow: Entrenar clasificadores con FEATURES ESTADÍSTICAS MÚLTIPLES + TRAIN/TEST.
Maneja archivos heterogéneos + validación robusta.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

from configs.config import settings
from src.models.classifier import AnomalyClassifier
from src.models.residuals_model import DataResidualsProcessor
from src.visualization.plots import COLORS


def extract_statistical_features(residuals: np.ndarray) -> np.ndarray:
    """
    Extrae MÚLTIPLES features estadísticas de residuos.
    """
    from scipy import stats

    abs_res = np.abs(residuals)
    flat_res = abs_res.flatten()

    features = {
        "mean": np.mean(flat_res),
        "max": np.max(flat_res),
        "std": np.std(flat_res),
        "p75": np.percentile(flat_res, 75),
        "p95": np.percentile(flat_res, 95),
        "range": np.max(flat_res) - np.min(flat_res),
        "cv": np.std(flat_res) / (np.mean(flat_res) + 1e-10),
        "kurtosis": stats.kurtosis(flat_res),
        "skewness": stats.skew(flat_res),
        "energy": np.sum(flat_res**2),
        "rms": np.sqrt(np.mean(flat_res**2)),
        "entropy": stats.entropy(
            np.histogram(flat_res, bins=20)[0] + 1e-10
        ),
    }

    return np.array([list(features.values())])


def create_comparison_plots(
    classifiers_results,
    output_dir: Path,
    y_test: np.ndarray,
):
    """Crea gráficas de comparación entre métodos (TRAIN vs TEST + ROC)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(classifiers_results.keys())

    # === Gráfica 1: Train vs Test Accuracy/AUC ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

    train_acc = [
        classifiers_results[m]["train_metrics"]["accuracy"] for m in methods
    ]
    test_acc = [
        classifiers_results[m]["test_metrics"]["accuracy"] for m in methods
    ]
    train_auc = [
        classifiers_results[m]["train_metrics"]["auc_roc"] for m in methods
    ]
    test_auc = [
        classifiers_results[m]["test_metrics"]["auc_roc"] for m in methods
    ]

    x = np.arange(len(methods))
    width = 0.35

    # Accuracy
    bars1 = ax1.bar(
        x - width / 2,
        train_acc,
        width,
        label="Train",
        color=COLORS["primary"],
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x + width / 2,
        test_acc,
        width,
        label="Test",
        color=COLORS["accent"],
        alpha=0.8,
    )

    ax1.set_xlabel("Método", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax1.set_title("Accuracy: Train vs Test", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.capitalize() for m in methods])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim([0, 1.05])

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # AUC-ROC
    bars3 = ax2.bar(
        x - width / 2,
        train_auc,
        width,
        label="Train",
        color=COLORS["primary"],
        alpha=0.8,
    )
    bars4 = ax2.bar(
        x + width / 2,
        test_auc,
        width,
        label="Test",
        color=COLORS["accent"],
        alpha=0.8,
    )

    ax2.set_xlabel("Método", fontsize=11, fontweight="bold")
    ax2.set_ylabel("AUC-ROC", fontsize=11, fontweight="bold")
    ax2.set_title("AUC-ROC: Train vs Test", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.capitalize() for m in methods])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim([0, 1.05])

    for bar in bars3 + bars4:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "01_train_vs_test.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # === Gráfica 2: Matrices de confusión (TEST) ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=100)

    for idx, method in enumerate(methods):
        cm = np.array(
            classifiers_results[method]["test_metrics"]["confusion_matrix"]
        )

        axes[idx].imshow(cm, cmap="Blues", aspect="auto")

        axes[idx].set_title(
            f"{method.capitalize()} (TEST)",
            fontsize=11,
            fontweight="bold",
        )
        axes[idx].set_xlabel("Predicho")
        axes[idx].set_ylabel("Real")

        for i in range(2):
            for j in range(2):
                axes[idx].text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12,
                    fontweight="bold",
                )

        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(["Desbal", "Desalin"])
        axes[idx].set_yticklabels(["Desbal", "Desalin"])

    plt.tight_layout()
    plt.savefig(
        output_dir / "02_confusion_matrices_test.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # === Gráfica 3: Curvas ROC (Test) ===
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

    color_map = {
        "linear": COLORS["primary"],
        "logistic": COLORS["accent"],
        "gmm": COLORS["warning"],
    }

    for method in methods:
        y_score = classifiers_results[method]["test_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            color=color_map.get(method, "gray"),
            lw=2,
            label=f"{method.upper()} (AUC = {roc_auc:.2f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Azar")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("Curvas ROC (Set de Prueba)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "04_roc_curves.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # === Gráfica 4: Overfitting (Acc gap) ===
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    overfit_gaps = [
        classifiers_results[m]["train_metrics"]["accuracy"]
        - classifiers_results[m]["test_metrics"]["accuracy"]
        for m in methods
    ]

    colors = [
        COLORS["warning"] if gap > 0.1 else COLORS["primary"]
        for gap in overfit_gaps
    ]
    bars = ax.bar(
        methods,
        overfit_gaps,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    ax.axhline(
        y=0.1,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Umbral overfitting (0.1)",
    )
    ax.set_xlabel("Método", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gap (Train - Test)", fontsize=11, fontweight="bold")
    ax.set_title("Detección de Overfitting", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, gap in zip(bars, overfit_gaps):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{gap:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "03_overfitting_detection.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("✓ Gráficas generadas (incluyendo ROC)")


def main():
    print("\n" + "=" * 70)
    print("🤖 ENTRENAMIENTO CLASIFICADORES (FEATURES + TRAIN/TEST SPLIT)")
    print("=" * 70 + "\n")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    run_name = (
        f"classifier_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        # Directorios
        mlartifacts_base = settings.ROOT_DIR / "mlartifacts"
        mlflow_artifacts_dir = (
            mlartifacts_base / str(experiment_id) / run_id / "artifacts"
        )
        classifiers_dir = mlflow_artifacts_dir / "classifiers"
        metrics_dir = mlflow_artifacts_dir / "metrics"
        plots_dir = mlflow_artifacts_dir / "plots"

        classifiers_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        project_models_dir = settings.ROOT_DIR / "models" / "trained"
        project_models_dir.mkdir(parents=True, exist_ok=True)

        print(f"📂 MLflow: {mlflow_artifacts_dir}")
        print(f"📂 Proyecto: {project_models_dir}\n")

        mlflow.set_tags(
            {
                "stage": "classifier_training",
                "n_features": 12,
                "methods": "linear,logistic,gmm",
                "validation": "train_test_split",
            }
        )

        # === PASO 1: CARGAR MODELO RESIDUOS ===
        print("📥 Cargando modelo de residuos base...\n")

        model_path = sorted(project_models_dir.glob("residuals_*.pkl"))[-1]
        residuals_model = DataResidualsProcessor.load(str(model_path))

        print(f"✓ Modelo: {model_path.name}\n")

        # === PASO 2: EXTRAER FEATURES DE TODOS LOS ARCHIVOS ===
        print("=" * 70)
        print("EXTRAYENDO FEATURES ESTADÍSTICAS")
        print("=" * 70 + "\n")

        X_features = []
        y_labels = []

        processed_dir = settings.DATA_DIR / "processed"
        
        # --- PROCESAR DESBALANCEO (Clase 0) ---
        files_0 = list((processed_dir / "imbalance").glob("*.csv"))
        print(f"📂 Procesando {len(files_0)} archivos de Desbalanceo...")
        
        for i, f in enumerate(files_0, 1):
            try:
                # Calcular residuos para este archivo
                # El método calculate_residuals_global maneja internamente
                # la selección de sensores que coincidan
                res_matrix, _, _, _, _ = (
                    residuals_model.calculate_residuals_global(
                        ruta_archivo=str(f)
                    )
                )
                
                # EXTRAER FEATURE (Media absoluta)
                # Esto convierte (n_samples, n_sensors) -> (1, 1)
                features = extract_statistical_features(res_matrix)
                X_features.append(features[0])  # ← Ahora es array de 12
                y_labels.append(0)

                if i % 5 == 0 or i == len(files_0):
                    print(f"  ✓ {i}/{len(files_0)} archivos procesados")
            except Exception as e:
                print(f"  ⚠️ {f.name}: {e}")

        # DESALINEACIÓN (1)
        files_1 = sorted(
            list((processed_dir / "misalignment").glob("*.csv"))
        )
        print(f"\n📂 DESALINEACIÓN: {len(files_1)} archivos\n")

        for i, f in enumerate(files_1, 1):
            try:
                res_matrix, _, _, _, _ = residuals_model.calculate_residuals_global(
                    ruta_archivo=str(f)
                )
                features = extract_statistical_features(res_matrix)
                X_features.append(features[0])
                y_labels.append(1)

                if i % 5 == 0 or i == len(files_1):
                    print(f"  ✓ {i}/{len(files_1)} archivos procesados")
            except Exception as e:
                print(f"  ⚠️ {f.name}: {e}")

        X = np.array(X_features)
        y = np.array(y_labels)

        print(f"\n{'=' * 70}")
        print("📊 DATASET COMPLETO")
        print(f"{'=' * 70}")
        print(f"X shape: {X.shape} (Archivos, Features)")
        print(f"y shape: {y.shape}")
        print(f"Desbalanceo (0): {np.sum(y == 0)} archivos")
        print(f"Desalineación (1): {np.sum(y == 1)} archivos\n")

        # === PASO 3: TRAIN/TEST SPLIT ===
        print("=" * 70)
        print("TRAIN/TEST SPLIT")
        print("=" * 70 + "\n")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        print(f"Total archivos: {len(X)}")
        print(f"Train: {len(X_train)} (80%)")
        print(f"Test: {len(X_test)} (20%)\n")
        print(
            f"Train - Desbalanceo: {np.sum(y_train==0)} | "
            f"Desalineación: {np.sum(y_train==1)}"
        )
        print(
            f"Test  - Desbalanceo: {np.sum(y_test==0)} | "
            f"Desalineación: {np.sum(y_test==1)}\n"
        )

        # === PASO 4: ENTRENAR MÉTODOS ===
        print("=" * 70)
        print("ENTRENANDO CLASIFICADORES")
        print("=" * 70 + "\n")

        methods = ["linear", "logistic", "gmm"]
        classifiers_results = {}

        for method in methods:
            print(f"\n🔧 Método: {method.upper()}")
            print("-" * 70)

            classifier = AnomalyClassifier(method=method)
            classifier.fit(X_train, y_train, verbose=True)

            train_metrics = classifier.evaluate(X_train, y_train)
            print("\n📈 Resultados TRAIN:")
            print(f"   Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"   AUC-ROC: {train_metrics['auc_roc']:.4f}")

            test_metrics = classifier.evaluate(X_test, y_test)
            print("\n📈 Resultados TEST:")
            print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"   AUC-ROC: {test_metrics['auc_roc']:.4f}")
            print(
                f"   Confusion Matrix:\n"
                f"{np.array(test_metrics['confusion_matrix'])}"
            )

            overfit_gap = (
                train_metrics["accuracy"] - test_metrics["accuracy"]
            )
            if overfit_gap > 0.15:
                print(f"\n⚠️ OVERFITTING DETECTADO (Gap: {overfit_gap:.4f})")
            elif overfit_gap > 0.05:
                print(f"\n⚡ Ligero overfitting (Gap: {overfit_gap:.4f})")
            else:
                print(f"\n✅ Generalización buena (Gap: {overfit_gap:.4f})")

            # Probabilidades para ROC
            y_proba_test = classifier.predict_proba(X_test)

            classifiers_results[method] = {
                "classifier": classifier,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "test_proba": y_proba_test,
            }

            mlflow.log_metrics(
                {
                    f"{method}_train_accuracy": train_metrics["accuracy"],
                    f"{method}_train_auc": train_metrics["auc_roc"],
                    f"{method}_test_accuracy": test_metrics["accuracy"],
                    f"{method}_test_auc": test_metrics["auc_roc"],
                    f"{method}_overfit_gap": overfit_gap,
                }
            )

            clf_path = classifiers_dir / f"classifier_{method}.pkl"
            classifier.save(str(clf_path))

            proj_clf_path = project_models_dir / f"classifier_{method}.pkl"
            classifier.save(str(proj_clf_path))

        # === SELECCIONAR MEJOR MODELO (por Test Accuracy) ===
        best_method = max(
            classifiers_results.items(),
            key=lambda x: x[1]["test_metrics"]["accuracy"],
        )[0]
        best_classifier = classifiers_results[best_method]["classifier"]

        best_path = project_models_dir / "classifier_best.pkl"
        best_classifier.save(str(best_path))

        # === GRÁFICAS ===
        print(f"\n{'=' * 70}")
        print("GENERANDO GRÁFICAS")
        print(f"{'=' * 70}\n")
        create_comparison_plots(classifiers_results, plots_dir, y_test)

        # === REPORTES ===
        comparison_report = {
            "timestamp": datetime.now().isoformat(),
            "n_features": 12,
            "feature_names": [
                "mean",
                "max",
                "std",
                "p75",
                "p95",
                "range",
                "cv",
                "kurtosis",
                "skewness",
                "energy",
                "rms",
                "entropy",
            ],
            "dataset": {
                "total_files": len(X),
                "desbalanceo_total": int(np.sum(y == 0)),
                "desalineacion_total": int(np.sum(y == 1)),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_desbalanceo": int(np.sum(y_train == 0)),
                "train_desalineacion": int(np.sum(y_train == 1)),
                "test_desbalanceo": int(np.sum(y_test == 0)),
                "test_desalineacion": int(np.sum(y_test == 1)),
            },
            "methods": {},
        }

        for method, result in classifiers_results.items():
            comparison_report["methods"][method] = {
                "train_accuracy": float(
                    result["train_metrics"]["accuracy"]
                ),
                "train_auc": float(result["train_metrics"]["auc_roc"]),
                "test_accuracy": float(
                    result["test_metrics"]["accuracy"]
                ),
                "test_auc": float(result["test_metrics"]["auc_roc"]),
                "overfit_gap": float(
                    result["train_metrics"]["accuracy"]
                    - result["test_metrics"]["accuracy"]
                ),
                "confusion_matrix_test": result["test_metrics"][
                    "confusion_matrix"
                ],
            }

        comparison_report["best_method"] = best_method
        comparison_report["best_test_accuracy"] = float(
            classifiers_results[best_method]["test_metrics"]["accuracy"]
        )

        report_path = metrics_dir / "classifier_comparison.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(comparison_report, f, indent=2)

        best_metadata = {
            "best_method": best_method,
            "train_accuracy": float(
                classifiers_results[best_method]["train_metrics"][
                    "accuracy"
                ]
            ),
            "test_accuracy": float(
                classifiers_results[best_method]["test_metrics"][
                    "accuracy"
                ]
            ),
            "test_auc": float(
                classifiers_results[best_method]["test_metrics"][
                    "auc_roc"
                ]
            ),
            "overfit_gap": float(
                classifiers_results[best_method]["train_metrics"][
                    "accuracy"
                ]
                - classifiers_results[best_method]["test_metrics"][
                    "accuracy"
                ]
            ),
            "timestamp": datetime.now().isoformat(),
        }

        with open(
            project_models_dir / "best_classifier_metadata.json", "w"
        ) as f:
            json.dump(best_metadata, f, indent=2)

        # === RESUMEN FINAL ===
        print("=" * 70)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("=" * 70)
        print("\n📊 RESUMEN POR MÉTODO (TEST)\n")

        for method in methods:
            train_acc = classifiers_results[method]["train_metrics"][
                "accuracy"
            ]
            test_acc = classifiers_results[method]["test_metrics"][
                "accuracy"
            ]
            test_auc = classifiers_results[method]["test_metrics"][
                "auc_roc"
            ]
            gap = train_acc - test_acc
            marker = "⭐" if method == best_method else "  "

            overfit_warning = (
                "⚠️ OVERFIT"
                if gap > 0.15
                else ("⚡ Ligero" if gap > 0.05 else "✅")
            )

            print(
                f"{marker} {method.upper():10s} | "
                f"Train: {train_acc:.4f} | "
                f"Test: {test_acc:.4f} | "
                f"AUC: {test_auc:.4f} | {overfit_warning}"
            )

        print(f"\n🏆 MEJOR MODELO (TEST): {best_method.upper()}")
        print(
            f"   Test Accuracy: "
            f"{classifiers_results[best_method]['test_metrics']['accuracy']:.4f}"
        )
        print(
            f"   Test AUC-ROC: "
            f"{classifiers_results[best_method]['test_metrics']['auc_roc']:.4f}"
        )
        print("   Guardado como: classifier_best.pkl\n")

        print("📁 Archivos guardados:")
        print("   - classifier_linear.pkl")
        print("   - classifier_logistic.pkl")
        print("   - classifier_gmm.pkl")
        print("   - classifier_best.pkl")
        print("   - best_classifier_metadata.json")
        print("   - classifier_comparison.json\n")

        print(
            f"🌐 MLflow: http://localhost:5000/#/experiments/"
            f"{experiment_id}/runs/{run_id}\n"
        )
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
