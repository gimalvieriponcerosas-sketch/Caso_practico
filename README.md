# Sistema interactivo de reconocimiento de dígitos

Prototipo de escritorio en Python que clasifica dígitos del 0 al 9 usando el conjunto `load_digits` de scikit-learn y una red neuronal de Keras. La aplicación permite dibujar con el ratón, visualizar la imagen 8×8 procesada, consultar probabilidades y guardar resultados en CSV.

## Requisitos

- Python 3.10 o posterior
- Tkinter (incluido en las distribuciones habituales de Python para Windows)
- Paquetes de `requirements.txt`

Instalación:

```powershell
python -m pip install -r requirements.txt
```

## Ejecución

1. Entrenar y evaluar las dos arquitecturas (solo al crear o regenerar el modelo):

   ```powershell
   python modulo_ml.py --entrenar
   ```

2. Abrir la aplicación de escritorio:

   ```powershell
   python modulo_ml.py
   ```

   Dibuja un dígito, indica la etiqueta que intentaste escribir y pulsa **Predecir**. Para contribuir a la evaluación de dibujos propios, marca la casilla correspondiente antes de predecir. El primer intento queda guardado; pulsa **Limpiar** para empezar el siguiente dibujo.

3. Al final, generar el informe de ruido y resumir los dibujos propios registrados:

   ```powershell
   python modulo_ml.py --analisis-final
   ```

Este análisis no reentrena ni modifica el modelo. La selección de las 30 imágenes y el ruido usan semillas fijas.

## Resultados incluidos

- `modelos_digitos/mejor_modelo.keras`: modelo persistido para abrir la aplicación sin volver a entrenar.
- `modelos_digitos/configuracion.json`: configuración, preprocesamiento y evaluación de prueba.
- `modelos_digitos/matriz_confusion.png`, `curvas_aprendizaje.png` y `errores_reales.png`: gráficos de entrenamiento y evaluación.
- `modelos_digitos/prueba_ruido.csv` e `informe_parte3.json`: resultados reproducibles de la prueba con ruido y resumen de dibujos propios.
- `modelos_digitos/dibujos_propios_v2.csv`: primeros intentos de dibujos manuscritos con la transformación actual. El archivo indica cuántos hay; no representa una prueba completa hasta reunir dos dibujos de cada dígito.

La evaluación final de prueba del modelo guardado fue 97,22 % de accuracy y 0,9719 de F1 macro. En una muestra fija de 30 imágenes, la accuracy fue 96,67 % sin ruido y 90 % con ruido gaussiano (desviación 0,15). Los dibujos de los usuarios pueden diferir del estilo de `load_digits`; se conservan también las predicciones incorrectas.

## Preprocesamiento del lienzo

El fondo se excluye con un umbral de intensidad, el trazo se recorta y se escala proporcionalmente hasta un máximo de 6×6 píxeles. Después se centra sobre fondo negro en una matriz de 8×8. No se invierten los colores: tanto el conjunto como el lienzo tienen trazo claro sobre fondo oscuro. Los 64 valores se llevan a [0, 1], equivalente a dividir entre 16 las intensidades originales de `load_digits`.

El umbral de aviso de baja confianza es 0,70 y es una regla didáctica; no demuestra que las probabilidades estén calibradas.
