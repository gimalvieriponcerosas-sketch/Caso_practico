"""Reconocimiento interactivo de dígitos manuscritos (scikit-learn + Keras).

Ejecuta `python modulo_ml.py`. La primera ejecución entrena y guarda los modelos
en `modelos_digitos/`; luego abre una pequeña interfaz Tkinter para dibujar.
"""

from __future__ import annotations

import json
import csv
import argparse
import os
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageTk
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
OUT = Path(__file__).resolve().parent / "modelos_digitos"
OUT.mkdir(exist_ok=True)
KERAS_FILE = OUT / "mejor_modelo.keras"
META_FILE = OUT / "configuracion.json"
CONFUSION_FILE = OUT / "matriz_confusion.png"
ERRORS_FILE = OUT / "errores_reales.png"
LEARNING_FILE = OUT / "curvas_aprendizaje.png"
HISTORY_FILE = OUT / "historial.csv"
OWN_DRAWINGS_FILE_ANTERIORES = OUT / "dibujos_propios.csv"
OWN_DRAWINGS_FILE = OUT / "dibujos_propios_v2.csv"
NOISE_RESULTS_FILE = OUT / "prueba_ruido.csv"
FINAL_REPORT_FILE = OUT / "informe_parte3.json"
TOLERANCIA = 1e-6
PREPROCESAMIENTO_APP = (
    "v2: recorte de tinta (>20), ajuste proporcional a un maximo de 6x6, "
    "centrado en lienzo negro 8x8; sin inversion de color; escala final 0..1."
)


def preparar_datos():
    """Separa 60/20/20, estratificado y con semilla fija; píxeles en [0, 1]."""
    digits = load_digits()
    x = digits.images.reshape(-1, 64).astype("float32") / 16.0
    y = digits.target.astype("int64")
    x_train, x_hold, y_train, y_hold = train_test_split(
        x, y, test_size=0.40, random_state=SEED, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_hold, y_hold, test_size=0.50, random_state=SEED, stratify=y_hold
    )
    return digits, (x_train, y_train, x_val, y_val, x_test, y_test)


def crear_secuencial():
    model = keras.Sequential([
        keras.Input(shape=(64,), name="pixeles"),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.15),
        layers.Dense(32, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ], name="red_secuencial")
    return model


def crear_funcional():
    inputs = keras.Input(shape=(64,), name="pixeles")
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="red_funcional")


def compilar(model):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def evaluar_y_guardar():
    keras.utils.set_random_seed(SEED)
    digits, data = preparar_datos()
    x_train, y_train, x_val, y_val, x_test, y_test = data
    constructors = [("secuencial", crear_secuencial), ("funcional", crear_funcional)]
    fits = {}
    for index, (name, constructor) in enumerate(constructors):
        keras.utils.set_random_seed(SEED + index)
        model = constructor()
        compilar(model)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True
        )
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                            epochs=150, batch_size=32, callbacks=[early_stop],
                            verbose=0)
        val_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        fits[name] = {"model": model, "history": history, "val_f1": float(val_f1)}

    # El conjunto de prueba no interviene en la selección ni en la configuración.
    chosen_name = max(fits, key=lambda key: fits[key]["val_f1"])
    chosen = fits[chosen_name]["model"]
    test_probs = chosen.predict(x_test, verbose=0)
    test_pred = np.argmax(test_probs, axis=1)
    test_f1 = float(f1_score(y_test, test_pred, average="macro"))
    test_accuracy = float(np.mean(test_pred == y_test))

    chosen.save(KERAS_FILE)
    metadata = {
        "modelo_elegido": chosen_name,
        "seleccion": "F1 macro de validacion",
        "semilla": SEED,
        "proporcion": {"entrenamiento": 0.60, "validacion": 0.20, "prueba": 0.20},
        "preprocesamiento": "Imagen 8x8; intensidades originales 0..16 divididas entre 16; vector de 64 valores.",
        "preprocesamiento_interfaz": PREPROCESAMIENTO_APP,
        "tolerancia_recarga": TOLERANCIA,
        "resultados": {
            name: {"f1_macro_validacion": item["val_f1"],
                  "parametros": int(item["model"].count_params()),
                  "epocas_efectivas": len(item["history"].history["loss"])}
            for name, item in fits.items()
        },
        "prueba_final": {"accuracy": test_accuracy, "f1_macro": test_f1},
    }
    META_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    matrix = confusion_matrix(y_test, test_pred, labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=np.arange(10)).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Matriz de confusión — {chosen_name}; F1 macro={test_f1:.3f}")
    fig.tight_layout()
    fig.savefig(CONFUSION_FILE, dpi=150)
    plt.close(fig)

    wrong = np.flatnonzero(test_pred != y_test)
    count = min(12, len(wrong))
    if count:
        fig, axes = plt.subplots(3, 4, figsize=(8, 6))
        for ax, index in zip(axes.flat, wrong[:count]):
            ax.imshow(x_test[index].reshape(8, 8) * 16, cmap="gray_r", vmin=0, vmax=16)
            ax.set_title(f"Real: {y_test[index]} | Pred: {test_pred[index]}")
            ax.axis("off")
        for ax in axes.flat[count:]:
            ax.axis("off")
        fig.suptitle("Errores reales del conjunto de prueba")
        fig.tight_layout()
        fig.savefig(ERRORS_FILE, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, item) in zip(axes, fits.items()):
        hist = item["history"].history
        ax.plot(hist["loss"], label="Entrenamiento")
        ax.plot(hist["val_loss"], label="Validación")
        ax.set_title(f"{name.capitalize()} — F1 val={item['val_f1']:.3f}")
        ax.set_xlabel("Época")
        ax.set_ylabel("Pérdida")
        ax.legend()
    fig.tight_layout()
    fig.savefig(LEARNING_FILE, dpi=150)
    plt.close(fig)

    # Verificación independiente de persistencia sobre cinco muestras fijas.
    reloaded = keras.models.load_model(KERAS_FILE)
    before = chosen.predict(x_test[:5], verbose=0)
    after = reloaded.predict(x_test[:5], verbose=0)
    max_diff = float(np.max(np.abs(before - after)))
    if max_diff > TOLERANCIA:
        raise RuntimeError(f"La recarga excedió la tolerancia: {max_diff:.3g} > {TOLERANCIA}")
    metadata["verificacion_recarga"] = {
        "muestras": 5, "diferencia_maxima": max_diff,
        "tolerancia": TOLERANCIA, "correcta": True,
    }
    META_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def cargar_o_entrenar():
    if KERAS_FILE.exists() and META_FILE.exists():
        try:
            metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
            return keras.models.load_model(KERAS_FILE), metadata
        except (OSError, ValueError, TypeError):
            pass
    metadata = evaluar_y_guardar()
    return keras.models.load_model(KERAS_FILE), metadata


def analizar_parte3():
    """Evalúa una muestra fija con ruido; solo carga el modelo ya congelado."""
    if not KERAS_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("Primero ejecuta: python modulo_ml.py --entrenar")
    model = keras.models.load_model(KERAS_FILE)
    _, (_, _, _, _, x_test, y_test) = preparar_datos()
    rng = np.random.default_rng(2026)
    chosen = rng.choice(len(x_test), size=30, replace=False)
    x = x_test[chosen]
    y = y_test[chosen]
    noise = np.random.default_rng(2027).normal(0.0, 0.15, size=x.shape).astype("float32")
    x_noisy = np.clip(x + noise, 0.0, 1.0)
    pred_original = np.argmax(model.predict(x, verbose=0), axis=1)
    pred_noise = np.argmax(model.predict(x_noisy, verbose=0), axis=1)
    acc_original = float(np.mean(pred_original == y))
    acc_noise = float(np.mean(pred_noise == y))
    with NOISE_RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["indice_test", "real", "pred_original", "correcto_original",
                         "pred_ruido", "correcto_ruido"])
        for idx, label, p0, p1 in zip(chosen, y, pred_original, pred_noise):
            writer.writerow([int(idx), int(label), int(p0), int(label == p0),
                             int(p1), int(label == p1)])

    own_rows = []
    if OWN_DRAWINGS_FILE.exists():
        with OWN_DRAWINGS_FILE.open(newline="", encoding="utf-8") as file:
            own_rows = list(csv.DictReader(file))
    previous_own_count = 0
    if OWN_DRAWINGS_FILE_ANTERIORES.exists():
        with OWN_DRAWINGS_FILE_ANTERIORES.open(newline="", encoding="utf-8") as file:
            previous_own_count = sum(1 for _ in csv.DictReader(file))
    own_summary = None
    if own_rows:
        correct = [row["resultado"] == "correcto" for row in own_rows]
        counts_by_digit = {str(d): sum(row["etiqueta_indicada"] == str(d) for row in own_rows)
                           for d in range(10)}
        own_summary = {
            "cantidad": len(own_rows),
            "accuracy": float(np.mean(correct)),
            "avisos_baja_confianza": sum(row["aviso_baja_confianza"] == "1" for row in own_rows),
            "por_digito": counts_by_digit,
            "completo_dos_por_digito": len(own_rows) == 20 and all(n == 2 for n in counts_by_digit.values()),
        }
    report = {
        "modelo_congelado": str(KERAS_FILE.name),
        "semilla_seleccion_30": 2026,
        "semilla_ruido": 2027,
        "ruido_gaussiano_sigma": 0.15,
        "version_preprocesamiento_interfaz": "v2",
        "muestra_ruido": 30,
        "accuracy_original_30": acc_original,
        "accuracy_con_ruido_30": acc_noise,
        "accuracy_prueba_completa": json.loads(META_FILE.read_text(encoding="utf-8"))["prueba_final"]["accuracy"],
        "dibujos_propios": own_summary,
        "intentos_transformacion_anterior_conservados": previous_own_count,
        "diagnostico": "Las exactitudes pueden diferir porque los dibujos del usuario y el ruido no "
                       "siguen exactamente la distribución de las imágenes de prueba. Una mejora "
                       "para otra versión sería entrenar con aumentos moderados de traslación, grosor y ruido.",
        "nota": "Análisis final; el modelo y las reglas no se reajustaron.",
    }
    FINAL_REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


class Aplicacion:
    def __init__(self, root, model, metadata):
        self.root = root
        self.model = model
        self.metadata = metadata
        self.root.title("Reconocimiento de dígitos")
        self.root.resizable(False, False)
        main = ttk.Frame(root, padding=12)
        main.grid(sticky="nsew")
        self.canvas = tk.Canvas(main, width=280, height=280, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, rowspan=9, padx=(0, 12))
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        ttk.Label(main, text="Imagen 8 × 8 recibida por la red:").grid(
            row=0, column=1, sticky="w", pady=(0, 4))
        self.preview = ttk.Label(main, text="(vacía)")
        self.preview.grid(row=1, column=1, sticky="w")
        ttk.Label(main, text=f"Modelo: {metadata['modelo_elegido']}  |  F1 prueba: "
                  f"{metadata['prueba_final']['f1_macro']:.3f}").grid(
                      row=2, column=1, sticky="w", pady=(8, 4))
        ttk.Label(main, text="Dígito que quisiste dibujar (0–9):").grid(
            row=3, column=1, sticky="w")
        self.label_entry = ttk.Entry(main, width=8)
        self.label_entry.grid(row=4, column=1, sticky="w", pady=(2, 4))
        self.own_test = tk.BooleanVar(value=False)
        ttk.Checkbutton(main, text="Incluir en mis 20 dibujos (primer intento)",
                        variable=self.own_test).grid(row=5, column=1, sticky="w")
        self.result = ttk.Label(main, text="Dibuja un dígito", font=("Segoe UI", 14))
        self.result.grid(row=6, column=1, sticky="w", pady=4)
        self.top3 = ttk.Label(main, text="Tres mayores: —", wraplength=340)
        self.top3.grid(row=7, column=1, sticky="w")
        self.chart = tk.Canvas(main, width=340, height=190, bg="white", highlightthickness=1,
                               highlightbackground="#bbbbbb")
        self.chart.grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 4))
        buttons = ttk.Frame(main)
        buttons.grid(row=9, column=0, columnspan=2, sticky="w", pady=6)
        self.predict_button = ttk.Button(buttons, text="Predecir", command=self.predict)
        self.predict_button.grid(row=0, column=0, padx=(0, 6))
        ttk.Button(buttons, text="Limpiar", command=self.clear).grid(row=0, column=1)
        ttk.Label(buttons, text=f"Historial CSV: {HISTORY_FILE}\nDibujos propios v2: {OWN_DRAWINGS_FILE}", wraplength=500).grid(
            row=0, column=2, padx=(12, 0), sticky="w")

    def paint(self, event):
        x, y = event.x, event.y
        radius = 8
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                                fill="white", outline="white")
        self.draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.preview.configure(image="", text="(vacía)")
        self.preview.image = None
        self.result.configure(text="Dibuja un dígito")
        self.top3.configure(text="Tres mayores: —")
        self.chart.delete("all")
        self.predict_button.configure(state="normal")

    def procesar_dibujo(self):
        """Recorta tinta, encaja en 6x6 y centra en el lienzo final de 8x8.

        El margen negro deja espacio alrededor, el ajuste proporcional evita
        deformar el trazo y la escala 0..255 a 0..1 coincide con 0..16 / 16.
        No se invierten colores: el conjunto y el lienzo tienen tinta clara
        sobre fondo oscuro.
        """
        arr = np.asarray(self.image, dtype=np.uint8)
        ys, xs = np.where(arr > 20)
        if len(xs) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        crop = self.image.crop((x0, y0, x1, y1))
        scale = min(6.0 / crop.width, 6.0 / crop.height)
        size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        glyph = crop.resize(size, Image.Resampling.LANCZOS)
        small = Image.new("L", (8, 8), 0)
        small.paste(glyph, ((8 - glyph.width) // 2, (8 - glyph.height) // 2))
        return small

    def registrar(self, etiqueta, prediccion, confianza):
        existe = HISTORY_FILE.exists()
        with HISTORY_FILE.open("a", newline="", encoding="utf-8") as archivo:
            writer = csv.writer(archivo)
            if not existe:
                writer.writerow(["fecha_hora", "etiqueta_indicada", "prediccion",
                                 "probabilidad_maxima", "resultado"])
            writer.writerow([datetime.now().astimezone().isoformat(timespec="seconds"),
                             etiqueta, prediccion, f"{confianza:.8f}",
                             "correcto" if etiqueta == prediccion else "incorrecto"])

    def registrar_dibujo_propio(self, etiqueta, prediccion, confianza):
        existe = OWN_DRAWINGS_FILE.exists()
        with OWN_DRAWINGS_FILE.open("a", newline="", encoding="utf-8") as archivo:
            writer = csv.writer(archivo)
            if not existe:
                writer.writerow(["fecha_hora", "etiqueta_indicada", "prediccion",
                                 "probabilidad_maxima", "aviso_baja_confianza", "resultado"])
            writer.writerow([datetime.now().astimezone().isoformat(timespec="seconds"),
                             etiqueta, prediccion, f"{confianza:.8f}", int(confianza < 0.70),
                             "correcto" if etiqueta == prediccion else "incorrecto"])

    def predict(self):
        small = self.procesar_dibujo()
        if small is None:
            messagebox.showwarning("Lienzo vacío", "Dibuja un dígito antes de reconocerlo.")
            return
        raw_label = self.label_entry.get().strip()
        if not raw_label.isdigit() or not 0 <= int(raw_label) <= 9:
            messagebox.showwarning("Etiqueta no válida", "Indica el dígito que quisiste dibujar (0 a 9).")
            return
        label = int(raw_label)
        if self.own_test.get() and OWN_DRAWINGS_FILE.exists():
            with OWN_DRAWINGS_FILE.open(newline="", encoding="utf-8") as archivo:
                count = sum(row["etiqueta_indicada"] == str(label)
                            for row in csv.DictReader(archivo))
            if count >= 2:
                messagebox.showwarning("Ya hay dos dibujos", f"Ya registraste dos dibujos del {label}.")
                return
        # 8x8 visibles exactamente como vector de 64 floats en escala [0, 1].
        x = np.asarray(small, dtype="float32").reshape(1, 64) / 255.0
        shown = small.resize((96, 96), Image.Resampling.NEAREST)
        self.preview_image = ImageTk.PhotoImage(shown)
        self.preview.configure(image=self.preview_image, text="")
        probabilities = self.model.predict(x, verbose=0)[0]
        guess = int(np.argmax(probabilities))
        confidence = float(probabilities[guess])
        self.result.configure(text=f"Predicción: {guess}  ({confidence:.1%})")
        top = np.argsort(probabilities)[-3:][::-1]
        self.top3.configure(text="Tres mayores: " + " | ".join(
            f"{digit}: {probabilities[digit]:.1%}" for digit in top
        ))
        self.chart.delete("all")
        self.chart.create_text(170, 12, text="Probabilidades por dígito", fill="#222")
        for digit, probability in enumerate(probabilities):
            y = 30 + digit * 16
            width = int(float(probability) * 230)
            color = "#dc6b32" if digit == guess else "#3979b8"
            self.chart.create_text(18, y + 5, text=str(digit), fill="#222")
            self.chart.create_rectangle(34, y, 34 + width, y + 11, fill=color, outline="")
            self.chart.create_text(295, y + 5, text=f"{probability:.1%}", fill="#222")
        if confidence < 0.70:
            messagebox.showwarning(
                "Baja confianza",
                "La mayor probabilidad es menor que 70%. La predicción puede ser incierta; "
                "este umbral didáctico no demuestra que las probabilidades estén calibradas."
            )
        self.registrar(label, guess, confidence)
        if self.own_test.get():
            self.registrar_dibujo_propio(label, guess, confidence)
        # Bloquear hasta limpiar garantiza que se conserve el primer intento.
        self.predict_button.configure(state="disabled")
def main():
    parser = argparse.ArgumentParser(description="Reconocimiento interactivo de dígitos")
    parser.add_argument("--entrenar", action="store_true",
                        help="entrena y evalúa los modelos (primera parte)")
    parser.add_argument("--analisis-final", action="store_true",
                        help="resume los 20 dibujos y evalúa 30 imágenes con ruido (tercera parte)")
    args = parser.parse_args()
    if args.analisis_final:
        try:
            report = analizar_parte3()
        except (FileNotFoundError, OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("Resultados detallados:", NOISE_RESULTS_FILE)
        print("Informe guardado:", FINAL_REPORT_FILE)
        return
    if args.entrenar:
        print("Entrenando y evaluando ambas redes...")
        metadata = evaluar_y_guardar()
        print("Prueba final:", metadata["prueba_final"])
        print("Verificación de recarga:", metadata["verificacion_recarga"])
        print("Modelo guardado en:", KERAS_FILE)
        return
    if not KERAS_FILE.exists() or not META_FILE.exists():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Modelo no encontrado", "No se encontró el modelo entrenado. Ejecuta primero la primera parte "
            "(entrenamiento) para generar modelos_digitos/mejor_modelo.keras."
        )
        root.destroy()
        return
    try:
        model = keras.models.load_model(KERAS_FILE)
        metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("No se pudo cargar el modelo", str(error))
        root.destroy()
        return
    metadata["preprocesamiento_interfaz"] = PREPROCESAMIENTO_APP
    META_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    root = tk.Tk()
    Aplicacion(root, model, metadata)
    root.mainloop()
if __name__ == "__main__":
    main()
