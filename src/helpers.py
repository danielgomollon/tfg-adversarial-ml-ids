"""
helpers.py  
Útiles de ayuda generales del proyecto TFG

=================================================
Funciones reutilizables por cualquier script o notebook.

  - set_seed              : reproducibilidad global
  - save_checkpoint       : guardado de estado completo de entrenamiento
  - load_checkpoint       : restauración de estado para resume tras crash
  - plot_training_curves  : curvas loss/F1/AUC por epoch
  - plot_pr_auc           : curva Precision-Recall para una clase
  - plot_confusion_matrix : matriz de confusión profesional
  - plot_attack_comparison: comparativa de métricas pre/post ataque
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from IPython.display import clear_output
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix

from src.config import Config


# ------------------------------------------------------------------
# REPRODUCIBILIDAD
# ------------------------------------------------------------------
def set_seed(seed=42):
    """
    Fija todas las semillas aleatorias para garantizar reproducibilidad.
    Obligatorio al principio de cada notebook o script.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[-] Semilla global fijada en {seed} (Reproducibilidad Garantizada).")


# ------------------------------------------------------------------
# CHECKPOINTS
# ------------------------------------------------------------------
def save_checkpoint(path, epoch, model, optimizer, scheduler, history, best_f1):
    """
    Guarda el estado completo del entrenamiento en disco.
    Permite reanudar exactamente donde se cortó (anti-crash Colab).

    Guarda: epoch, pesos del modelo, estado del optimizer,
            estado del scheduler, historial de métricas y mejor F1.
    """
    torch.save({
        'epoch'     : epoch,
        'model'     : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'history'   : history,
        'best_f1'   : best_f1,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    """
    Restaura el estado completo desde un checkpoint guardado.
    Devuelve (start_epoch, history, best_f1).

    Uso típico tras crash de Colab:
        start_epoch, history, best_f1 = load_checkpoint(
            path_last, model, optimizer, scheduler, device
        )
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    scheduler.load_state_dict(ck['scheduler'])
    print(f"   [→] Checkpoint restaurado — epoch {ck['epoch']} "
          f"(mejor F1: {ck['best_f1']:.4f})")
    return ck['epoch'], ck['history'], ck['best_f1']


# ------------------------------------------------------------------
# PLOTS DE ENTRENAMIENTO
# ------------------------------------------------------------------
def create_progress_bar(epochs, start_epoch=1, model_name=Config.MODEL_NAME):
    return tqdm(
        range(start_epoch, epochs + 1),
        desc          = f"{model_name}",
        unit          = "epoch",
        initial       = start_epoch - 1,
        total         = epochs,
        dynamic_ncols = True,
    )

def create_batch_progress_bar(loader, epoch, total_epochs):
    return tqdm(
        loader,
        desc          = f"  Epoch {epoch:03d}/{total_epochs}",
        leave         = False,
        unit          = "batch",
        dynamic_ncols = True,
    )

def print_epoch_summary(epoch, total_epochs, history):
    """Dashboard visual que se actualiza en tiempo real."""
    clear_output(wait=True)

    train_losses = history['train_loss']
    val_losses   = history['val_loss']
    val_f1s      = history['val_f1']
    val_aucs     = history['val_auc']

    # tendencia F1
    if len(val_f1s) >= 2:
        delta = val_f1s[-1] - val_f1s[-2]
        trend = f"▲ +{delta:.4f}" if delta > 0 else f"▼ {delta:.4f}"
        trend_color = "green" if delta > 0 else "red"
    else:
        trend = "—"
        trend_color = "gray"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        f"🧠 TabularResNet — Epoch {epoch}/{total_epochs}  |  "
        f"F1: {val_f1s[-1]:.4f}  |  AUC: {val_aucs[-1]:.4f}  |  "
        f"Tendencia: {trend}",
        fontsize=13, fontweight='bold'
    )

    epochs_range = range(1, len(train_losses) + 1)

    # Loss
    axes[0].plot(epochs_range, train_losses, 'o-', color='steelblue',
                 label='Train Loss', linewidth=2, markersize=4)
    axes[0].plot(epochs_range, val_losses,   'o-', color='darkorange',
                 label='Val Loss',   linewidth=2, markersize=4)
    axes[0].set_title('Cross-Entropy Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1
    axes[1].plot(epochs_range, val_f1s, 'o-', color='seagreen',
                 linewidth=2, markersize=4, label='Val F1-macro')
    axes[1].axhline(y=max(val_f1s), color='seagreen', linestyle='--',
                    alpha=0.5, label=f'Best: {max(val_f1s):.4f}')
    axes[1].set_title('F1-macro (Validación)', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim([0.5, 1.0])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC
    axes[2].plot(epochs_range, val_aucs, 'o-', color='mediumpurple',
                 linewidth=2, markersize=4, label='Val AUC-ROC')
    axes[2].axhline(y=max(val_aucs), color='mediumpurple', linestyle='--',
                    alpha=0.5, label=f'Best: {max(val_aucs):.4f}')
    axes[2].set_title('AUC-ROC (Validación)', fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim([0.9, 1.0])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # resumen textual debajo de la gráfica
    print(f"  Epoch {epoch:03d}/{total_epochs} | "
          f"T.Loss: {train_losses[-1]:.4f} | "
          f"V.Loss: {val_losses[-1]:.4f} | "
          f"F1: {val_f1s[-1]:.4f} | "
          f"AUC: {val_aucs[-1]:.4f}")

def plot_training_curves(history, save_path=None):
    """Curvas de entrenamiento para análisis post-entreno en notebooks."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train', color='steelblue')
    axes[0].plot(epochs, history['val_loss'],   label='Val',   color='darkorange')
    axes[0].set_title('Cross-Entropy Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['val_f1'], label='Val F1-macro', color='seagreen')
    axes[1].set_title('F1-macro (Validación)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['val_auc'], label='Val AUC-ROC', color='mediumpurple')
    axes[2].set_title('AUC-ROC (Validación)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim([0, 1])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------------
# PLOTS DE MÉTRICAS
# ------------------------------------------------------------------
def plot_pr_auc(y_true, y_probs, class_name="Ataque", save_path=None):
    """
    Dibuja la curva Precision-Recall para una clase binaria.
    Métrica fundamental para datasets desbalanceados en ciberseguridad.

    Parámetros:
        y_true     : etiquetas binarias reales (0/1)
        y_probs    : probabilidades predichas para la clase positiva
        class_name : nombre de la clase para el título
        save_path  : si se proporciona, guarda la figura en disco
    """
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall — {class_name}')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   [→] PR curve guardada en {save_path}")

    plt.show()
    return pr_auc


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Genera una matriz de confusión normalizada con estilo profesional.
    Usada en notebooks 02, 04 y 05 para comparar modelos.

    Parámetros:
        y_true      : etiquetas reales
        y_pred      : predicciones del modelo
        class_names : lista de nombres de clases (Config.CLASS_NAMES)
        save_path   : si se proporciona, guarda la figura en disco
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.ylabel('Clase Real',       fontweight='bold')
    plt.xlabel('Predicción',       fontweight='bold')
    plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   [→] Matriz guardada en {save_path}")

    plt.show()


# ------------------------------------------------------------------
# PLOTS DE ATAQUES ADVERSARIALES
# ------------------------------------------------------------------
def plot_attack_comparison(metrics_before, metrics_after, attack_name, save_path=None):
    """
    Barras comparativas de métricas antes y después de un ataque adversarial.
    Usada en notebook 04 para mostrar el impacto de FGSM/PGD/DeepFool.

    Parámetros:
        metrics_before : dict con {'accuracy', 'f1_macro', 'auc'}
        metrics_after  : dict con {'accuracy', 'f1_macro', 'auc'}
        attack_name    : nombre del ataque (e.g. 'FGSM ε=0.1')
        save_path      : si se proporciona, guarda la figura en disco
    """
    labels  = ['Accuracy', 'F1-macro', 'AUC-ROC']
    before  = [metrics_before['accuracy'],
               metrics_before['f1_macro'],
               metrics_before['auc']]
    after   = [metrics_after['accuracy'],
               metrics_after['f1_macro'],
               metrics_after['auc']]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, before, width, label='Sin ataque',
                   color='steelblue',  alpha=0.85)
    bars2 = ax.bar(x + width/2, after,  width, label=f'Tras {attack_name}',
                   color='crimson', alpha=0.85)

    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Métrica')
    ax.set_title(f'Impacto del ataque {attack_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Valores encima de cada barra
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   [→] Comparativa guardada en {save_path}")

    plt.show()