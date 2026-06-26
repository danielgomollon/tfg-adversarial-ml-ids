"""
src/models/resnet/prepare_refat_data.py
================================================================
Randomized Entropy Fast Adversarial Training (RE-FAT)
Generador de Dataset Sintético con ataques S3M.

Este script genera el pool de troyanos físicos necesarios para 
el re-entrenamiento adversarial de la ResNet en Colab.

Justificación:
  ratio_range alto (0.7-0.9) genera troyanos casi completamente benignos
  que confunden la frontera de decisión — el modelo aprende que
  "features benignas puede ser ataque", lo que destruye el recall
  en clases raras como Recon y Malware.
  
  El rango (0.3, 0.65) genera troyanos que son claramente una mezcla
  de ataque y benigno — suficiente camuflaje para ser difíciles de
  clasificar, pero con firma de ataque preservada.

"""

import os
import numpy as np
from src.utils.domain_constraints import DomainConstraints
from src.attacks.s3m_attack import generate_s3m_augmentation
from src.config import Config


def main():
    print("\n" + "="*60)
    print("S3M DATA PREPARATION v2 — ratio_range corregido")
    print("="*60)

    data_path     = Config.DATA_PROCESSED_PATH
    processed_dir = os.path.dirname(data_path)
    output_dir    = os.path.join(processed_dir, "ataques_s3m")
    os.makedirs(output_dir, exist_ok=True)

    print("[-] 1. Cargando DomainConstraints y datos...")
    dc      = DomainConstraints.from_artifacts()
    X_train = np.load(os.path.join(data_path, "X_train.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))

    idx_atk = np.where(y_train != 0)[0]
    idx_ben = np.where(y_train == 0)[0]

    print(f"   Train total  : {len(X_train):,}")
    print(f"   Ataques      : {len(idx_atk):,}")
    print(f"   Benignos     : {len(idx_ben):,}")

    # Distribución de clases de ataque para verificar balance
    unique, counts = np.unique(y_train[idx_atk], return_counts=True)
    print("\n   Distribución ataques en train:")
    class_names = {
        1:'DoS', 2:'DDoS', 3:'Web/Injection', 4:'Brute Force',
        5:'Recon', 6:'Malware', 7:'Exploits'
    }
    for cls, cnt in zip(unique, counts):
        print(f"     {class_names.get(int(cls), cls):<15} {cnt:>7,}")

    n_target = 100_000
    print(f"\n[-] 2. Generando {n_target:,} troyanos S3M...")
    print(f"   ratio_range: (0.3, 0.65)  ← CORREGIDO (era 0.2-0.9)")

    X_s3m, y_s3m = generate_s3m_augmentation(
        X_train_atk_sc = X_train[idx_atk],
        X_train_ben_sc = X_train[idx_ben],
        y_train_atk    = y_train[idx_atk],
        dc             = dc,
        n_augmented    = n_target,
        ratio_range    = (0.3, 0.65),  # CORREGIDO
        seed           = 42,
    )

    # Verificar balance de clases en los troyanos
    unique_s3m, counts_s3m = np.unique(y_s3m, return_counts=True)
    print("\n   Distribución troyanos generados:")
    for cls, cnt in zip(unique_s3m, counts_s3m):
        print(f"     {class_names.get(int(cls), cls):<15} {cnt:>7,}")

    output_x = os.path.join(output_dir, "X_train_s3m.npy")
    output_y = os.path.join(output_dir, "y_train_s3m.npy")

    print(f"\n[-] 3. Guardando artefactos...")
    np.save(output_x, X_s3m)
    np.save(output_y, y_s3m)

    print(f"\n[✓] COMPLETADO")
    print(f"   X: {output_x}  shape={X_s3m.shape}")
    print(f"   y: {output_y}  shape={y_s3m.shape}")
    print(f"\n   Siguiente paso: ejecutar REFAT_Trainer().run()")
    print("="*60)


if __name__ == "__main__":
    main()