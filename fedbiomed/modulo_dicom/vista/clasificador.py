import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from tkinter import filedialog, messagebox


# ============================================================================
# 1. MODELO FLEXIBLE CON TRANSFER LEARNING
# ============================================================================

class FlexibleMammographyClassifier(nn.Module):
    """
    Clasificador flexible que adapta la capa de entrada al número de características
    Mantiene los pesos pre-entrenados en las capas internas
    """

    def __init__(self, pretrained_path, input_size):
        super(FlexibleMammographyClassifier, self).__init__()

        # Cargar checkpoint pre-entrenado
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        self.input_size = input_size
        self.original_input_size = 518  # Ajustado al tamaño real de entrenamiento

        print(f"\n🔧 Configurando modelo:")
        print(f"   • Tamaño de entrada original: {self.original_input_size}")
        print(f"   • Tamaño de entrada requerido: {input_size}")

        # CASO 1: Input size coincide - usar modelo completo pre-entrenado
        if input_size == self.original_input_size:
            print(f"   ✅ Tamaño perfecto - usando modelo pre-entrenado completo")

            self.fc1 = nn.Linear(518, 16)
            self.fc1.weight.data = checkpoint['fc1.weight']
            self.fc1.bias.data = checkpoint['fc1.bias']

            self.adaptation_layer = None

        # CASO 2: Input size diferente - agregar capa de adaptación
        else:
            print(f"   🔄 Agregando capa de adaptación: {input_size} → 518")
            print(f"   ⚠️  Nota: La capa de adaptación se inicializa aleatoriamente")

            # Capa de adaptación (nueva, con pesos aleatorios)
            self.adaptation_layer = nn.Linear(input_size, 518)

            # Inicialización Xavier para la nueva capa
            nn.init.xavier_uniform_(self.adaptation_layer.weight)
            nn.init.zeros_(self.adaptation_layer.bias)

            # Capa fc1 original (pre-entrenada)
            self.fc1 = nn.Linear(518, 16)
            self.fc1.weight.data = checkpoint['fc1.weight']
            self.fc1.bias.data = checkpoint['fc1.bias']

        # Capa fc2 (siempre pre-entrenada)
        self.fc2 = nn.Linear(16, 2)
        self.fc2.weight.data = checkpoint['fc2.weight']
        self.fc2.bias.data = checkpoint['fc2.bias']

        # Activaciones
        self.relu = nn.ReLU()

        print(f"\n✅ Modelo configurado:")
        if self.adaptation_layer:
            print(f"   Arquitectura: {input_size} → 518 → 16 → 2")
            print(f"   Capas pre-entrenadas: fc1, fc2")
            print(f"   Capas nuevas: adaptation_layer")
        else:
            print(f"   Arquitectura: 518 → 16 → 2")
            print(f"   Todas las capas pre-entrenadas")

    def forward(self, x):
        """Propagación hacia adelante"""
        # Si hay capa de adaptación, aplicarla primero
        if self.adaptation_layer is not None:
            x = self.adaptation_layer(x)
            x = self.relu(x)

        # Capas pre-entrenadas
        x = self.fc1(x)
        x = self.relu(x)
        features = x  # Características de 16 dimensiones
        output = self.fc2(x)

        return output, features

    def predict(self, x):
        """Realizar predicción"""
        self.eval()
        with torch.no_grad():
            outputs, features = self.forward(x)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities, features


# ============================================================================
# 2. CARGAR DATASET
# ============================================================================

def load_dataset(csv_path):
    """
    Cargar dataset desde CSV (detecta automáticamente el número de características)
    """
    print(f"\n📂 Cargando dataset desde: {csv_path}")

    # Leer CSV
    df = pd.read_csv(csv_path)

    print(f"\n📊 Información del dataset:")
    print(f"   • Total de muestras: {len(df)}")
    print(f"   • Total de características: {len(df.columns)}")

    # Verificar si hay columna de diagnóstico
    diagnostic_cols = ['diagnostic', 'diagnosis', 'label', 'class', 'target']
    has_labels = False
    label_col = None

    for col in diagnostic_cols:
        if col in df.columns:
            has_labels = True
            label_col = col
            print(f"   ⚠️  Se detectó columna de etiquetas: '{col}'")
            print(f"      Esta columna será IGNORADA para la predicción")
            break

    # Separar características
    if has_labels:
        X = df.drop(columns=[label_col]).values
        feature_names = df.drop(columns=[label_col]).columns.tolist()
        y_true = df[label_col].values
    else:
        X = df.values
        feature_names = df.columns.tolist()
        y_true = None

    num_features = X.shape[1]
    print(f"\n✅ Características a usar: {num_features}")

    if num_features < 100:
        print(f"   ⚠️  Advertencia: Solo {num_features} características detectadas")
        print(f"      ¿Es correcto? Dataset multimodal típicamente tiene >512")

    # Convertir a tensor
    X_tensor = torch.FloatTensor(X)

    print(f"\n✅ Dataset preparado:")
    print(f"   • Shape: {X_tensor.shape}")
    print(f"   • Rango: [{X_tensor.min():.3f}, {X_tensor.max():.3f}]")

    return X_tensor, feature_names, df, y_true, has_labels


# ============================================================================
# 3. REALIZAR PREDICCIONES
# ============================================================================

def make_predictions(model, X):
    """Realizar predicciones"""
    print(f"\n🔍 Realizando predicciones sobre {len(X)} muestras...")

    predictions, probabilities, features = model.predict(X)

    # Convertir a numpy
    predictions = predictions.numpy()
    probabilities = probabilities.numpy()
    features = features.numpy()

    class_names = ['Benigno', 'Maligno']

    # Estadísticas
    unique, counts = np.unique(predictions, return_counts=True)

    print(f"\n📊 Distribución de predicciones:")
    for cls, count in zip(unique, counts):
        percentage = (count / len(predictions)) * 100
        print(f"   • {class_names[cls]}: {count} ({percentage:.1f}%)")

    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'features': features,
        'class_names': class_names
    }


# ============================================================================
# 4. VISUALIZACIONES
# ============================================================================

def plot_predictions(results, save_path='predicciones_visualizadas.png'):
    """Crear visualizaciones"""
    fig = plt.figure(figsize=(16, 10))

    predictions = results['predictions']
    probabilities = results['probabilities']
    features = results['features']
    class_names = results['class_names']

    # 1. Distribución de Predicciones
    ax1 = plt.subplot(2, 3, 1)
    unique, counts = np.unique(predictions, return_counts=True)
    colors = ['blue', 'red']
    bars = ax1.bar([class_names[i] for i in unique], counts,
                   color=[colors[i] for i in unique], alpha=0.7, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}\n({height / len(predictions) * 100:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Número de muestras', fontsize=11)
    ax1.set_title('Distribución de Predicciones', fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Distribución de Confianza
    ax2 = plt.subplot(2, 3, 2)
    for i, (label, color) in enumerate(zip(class_names, colors)):
        mask = predictions == i
        if mask.sum() > 0:
            probs = probabilities[mask, i]
            ax2.hist(probs, bins=20, alpha=0.6, label=label,
                     edgecolor='black', color=color)
    ax2.set_xlabel('Confianza de la predicción', fontsize=11)
    ax2.set_ylabel('Frecuencia', fontsize=11)
    ax2.set_title('Distribución de Confianza', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Probabilidad de Maligno
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(probabilities[:, 1], bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Umbral 0.5')
    ax3.set_xlabel('Probabilidad de Maligno', fontsize=11)
    ax3.set_ylabel('Frecuencia', fontsize=11)
    ax3.set_title('Distribución de P(Maligno)', fontsize=14, fontweight='bold', pad=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. PCA de Características
    ax4 = plt.subplot(2, 3, 4)
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    colors_map = ['blue', 'red']
    markers = ['o', '^']
    for i, (label, color, marker) in enumerate(zip(class_names, colors_map, markers)):
        mask = predictions == i
        if mask.sum() > 0:
            ax4.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=color, alpha=0.6, s=50, label=label,
                        marker=marker, edgecolors='black', linewidth=0.5)

    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax4.set_title('Espacio de Características', fontsize=14, fontweight='bold', pad=10)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. Análisis de Confianza
    ax5 = plt.subplot(2, 3, 5)
    max_conf = np.max(probabilities, axis=1)

    confidence_ranges = ['<0.5', '0.5-0.7', '0.7-0.9', '>0.9']
    counts = [
        (max_conf < 0.5).sum(),
        ((max_conf >= 0.5) & (max_conf < 0.7)).sum(),
        ((max_conf >= 0.7) & (max_conf < 0.9)).sum(),
        (max_conf >= 0.9).sum()
    ]

    bars = ax5.bar(confidence_ranges, counts,
                   color=['red', 'orange', 'yellowgreen', 'green'],
                   alpha=0.7, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax5.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}\n({height / len(predictions) * 100:.1f}%)',
                     ha='center', va='bottom', fontsize=9)

    ax5.set_ylabel('Número de muestras', fontsize=11)
    ax5.set_xlabel('Rango de confianza', fontsize=11)
    ax5.set_title('Distribución por Nivel de Confianza', fontsize=14, fontweight='bold', pad=10)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Estadísticas Generales
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    stats_text = f"""
╔═══════════════════════════════════════╗
║      ESTADÍSTICAS GENERALES           ║
╠═══════════════════════════════════════╣
║                                       ║
║  Total de muestras: {len(predictions):>6d}          ║
║                                       ║
║  ─────────────────────────────────    ║
║  PREDICCIONES:                        ║
║  ─────────────────────────────────    ║
"""

    for i, class_name in enumerate(class_names):
        mask = predictions == i
        count = mask.sum()
        avg_conf = probabilities[mask, i].mean() if count > 0 else 0
        stats_text += f"║  {class_name:<10s}: {count:>4d} ({100 * count / len(predictions):>5.1f}%)     ║\n"
        stats_text += f"║    Conf. prom.: {avg_conf:>6.1%}             ║\n"

    stats_text += f"""║                                       ║
║  ─────────────────────────────────    ║
║  CONFIANZA:                           ║
║  ─────────────────────────────────    ║
║  Media:        {max_conf.mean():>6.1%}              ║
║  Mediana:      {np.median(max_conf):>6.1%}              ║
║  Mín:          {max_conf.min():>6.1%}              ║
║  Máx:          {max_conf.max():>6.1%}              ║
║                                       ║
║  Alta (>90%):  {(max_conf > 0.9).sum():>4d} ({100 * (max_conf > 0.9).sum() / len(predictions):>5.1f}%)     ║
║  Baja (<60%):  {(max_conf < 0.6).sum():>4d} ({100 * (max_conf < 0.6).sum() / len(predictions):>5.1f}%)     ║
║                                       ║
╚═══════════════════════════════════════╝
"""

    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
                                                   facecolor='wheat', alpha=0.3))

    plt.suptitle('📊 Análisis de Predicciones',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualización guardada: {save_path}")
    plt.show()


# ============================================================================
# 5. EXPORTAR RESULTADOS
# ============================================================================

def export_results(results, df_original, output_path='resultados_predicciones.csv'):
    """Exportar predicciones a CSV"""
    predictions = results['predictions']
    probabilities = results['probabilities']
    class_names = results['class_names']

    # Crear DataFrame
    results_df = df_original.copy()
    results_df['prediccion'] = [class_names[p] for p in predictions]
    results_df['prediccion_codigo'] = predictions
    results_df['prob_benigno'] = probabilities[:, 0]
    results_df['prob_maligno'] = probabilities[:, 1]
    results_df['confianza'] = np.max(probabilities, axis=1)

    # Ordenar por confianza
    results_df = results_df.sort_values('confianza', ascending=False)

    # Guardar
    results_df.to_csv(output_path, index=False)
    print(f"✅ Resultados exportados: {output_path}")

    return results_df


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🔬 CLASIFICADOR FLEXIBLE DE MAMOGRAFÍAS")
    print("   (Soporta cualquier número de características)")
    print("=" * 70)

    # ========== CONFIGURACIÓN ==========
    MODEL_PATH = r'D:\Proyectos Python\modelo_federado.pth'
    CSV_PATH = r'D:\dataset_no_diagnostic.csv'  # TU ARCHIVO
    # ===================================

    # 1. Cargar dataset
    try:
        X, feature_names, df_original, y_true, has_labels = load_dataset(CSV_PATH)
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró {CSV_PATH}")
        return
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return

    # 2. Crear modelo adaptativo
    print("\n📦 Cargando modelo...")
    input_size = X.shape[1]
    model = FlexibleMammographyClassifier(MODEL_PATH, input_size)

    # 3. Predicciones
    results = make_predictions(model, X)

    # 4. Análisis
    print("\n" + "=" * 70)
    print("📈 ANÁLISIS DE CONFIANZA")
    print("=" * 70)

    max_conf = np.max(results['probabilities'], axis=1)
    print(f"\n   Confianza promedio: {max_conf.mean():.1%}")
    print(f"   Alta confianza (>90%): {(max_conf > 0.9).sum()} muestras")
    print(f"   Baja confianza (<60%): {(max_conf < 0.6).sum()} muestras")

    # 5. Visualizar
    print("\n🎨 Generando visualizaciones...")
    plot_predictions(results)

    # 6. Exportar
    print("\n💾 Exportando resultados...")
    results_df = export_results(results, df_original)

    print("\n📋 Primeras 10 predicciones:")
    print(results_df[['prediccion', 'prob_benigno', 'prob_maligno', 'confianza']].head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("✨ COMPLETADO")
    print("=" * 70)
    print("\n📁 Archivos generados:")
    print("   • predicciones_visualizadas.png")
    print("   • resultados_predicciones.csv")
    print("=" * 70)

def evaluar_estudio_cancer():
    # Seleccionar archivo CSV
    ruta_csv = "fbm-soli-node/data/resultados_modulo_dicom/dataset_cancer_mama_validacion.csv"
    if not ruta_csv:
        return

    try:
        # Cargar dataset
        X, _, _, _, _ = load_dataset(ruta_csv)

        # Cargar modelo
        MODEL_PATH = "fedbiomed/fedbiomed/modulo_dicom/vista/modelo/modelo_federado.pth"  # Ajusta si es necesario
        input_size = X.shape[1]
        model = FlexibleMammographyClassifier(MODEL_PATH, input_size)

        # Realizar predicción
        results = make_predictions(model, X)
        predicciones = results['predictions']
        class_names = results['class_names']

        # Determinar si hay al menos una muestra maligna
        tiene_cancer = (predicciones == 1).any()
        mensaje = "⚠️ El estudio presenta indicios de cáncer (Maligno)" if tiene_cancer else "✅ El estudio no presenta indicios de cáncer (Benigno)"

        messagebox.showinfo("Resultado del estudio", mensaje)

    except Exception as e:
        messagebox.showerror("Error", f"No se pudo evaluar el estudio:\n{e}")

if __name__ == "__main__":
    main()