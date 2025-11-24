import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import io
import base64

# ==============================================================================
# --- CONFIGURACIÓN Y MODELO ---
# ==============================================================================

# RUTAS RELATIVAS (Asegúrate de colocar los archivos en la misma carpeta del script)
# NOTA: Estas rutas deben ser actualizadas por el usuario.
MODEL_PATH = "best_cnn_97plus_model.keras"
LOGO_PATH = "logo.jpg" 

# Asegúrate que este orden coincide con el índice de salida de tu modelo
CLASS_NAMES_RAW = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'] 

# Mapeo para nombres amigables en la interfaz
CLASS_NAMES_MAP = {
    'NonDemented': "No Demencia (ND)",
    'VeryMildDemented': "Muy Leve (VMD)",
    'MildDemented': "Leve (MD)",
    'ModerateDemented': "Moderada (MOD)"
}
MODEL_CLASS_ORDER = CLASS_NAMES_RAW

IMG_HEIGHT = 128
IMG_WIDTH = 128

# --- PALETA DE COLORES ---
BG_WINDOW = "#EAF4FD"      
CARD_BG = "#FFFFFF"        
PRIMARY_BLUE = "#3B8ED4"   
SECONDARY_BLUE = "#549DDF" 
TEXT_DARK = "#2C3E50"      
TITLE_BLUE = "#1F487B"     
TEXT_LIGHT = "#FFFFFF"     
SUCCESS_COLOR = "#6e90ad"  # Verde para No Demencia
WARNING_COLOR = "#21568a"  # Rojo para Demencia
NEUTRAL_COLOR = "#F0F0F0"  
PROGRESS_BG = "#E0E0E0"    
PROGRESS_FG = PRIMARY_BLUE 

# --- FUENTES ---
TITLE_FONT = ("Segoe UI", 24, "bold") 
HEADER_FONT = ("Segoe UI", 14)       
BUTTON_FONT = ("Segoe UI", 16, "bold")
BODY_FONT = ("Segoe UI", 12)
RESULT_FONT = ("Segoe UI", 15, "bold") 
ICON_FONT = ("Segoe UI", 22) 
CLASS_BAR_FONT = ("Segoe UI", 10, "bold") 

# --- PLACEHOLDER DE LOGO (Base64 de una imagen simple 1x1) ---
# Usamos un logo simple si el archivo 'logo.jpg' no se encuentra.
PLACEHOLDER_LOGO_BASE64 = b'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAzUExURUdwDNv8/dn8/d78/Nf8/N38/Nz8/N78/N78/N78/d78/N78/N78/N78/d78/N78/d78/d78/d4s/gQAAAAQdFJOUwAQIDBAUGBweICQoLDQ4PEyI21wAAAAiSURBVDjLY2AYAAUYEcDAwMDACc/AwaADwACrM8CgAIAASJc21k9X+e8AAAAASUVORK5CYII='


# --- CARGA DEL MODELO (Con manejo de error y mock) ---
model = None
try:
    if os.path.exists(MODEL_PATH):
        # Desactivamos la compilación si el modelo ya está entrenado (ahorra tiempo y evita warnings)
        model = keras.models.load_model(MODEL_PATH, compile=False) 
        print("Modelo Keras cargado exitosamente.")
    else:
        print(f"ADVERTENCIA: No se encontró el modelo en '{MODEL_PATH}'. Usando modelo MOCK.")
except Exception as e:
    print(f"ERROR: No se pudo cargar el modelo Keras. Usando modelo MOCK. Detalle: {e}")

# Si el modelo no se cargó, creamos una clase simulada
if model is None:
    class MockModel:
        """Simula las predicciones de un modelo CNN para fines de prueba de la GUI."""
        def predict(self, img_tensor, verbose=0):
            # Simular una predicción (return probabilidades para las 4 clases)
            # Genera probabilidades aleatorias con un sesgo hacia 'NonDemented'
            if np.random.rand() < 0.7:
                # Caso No Demencia (Clase 2: NonDemented)
                probs = np.array([0.05, 0.05, 0.85, 0.05])
            else:
                # Caso Demencia (Distribución aleatoria entre los demás)
                weights = np.array([0.25, 0.25, 0.0, 0.5])
                probs = np.random.dirichlet(weights * 10)
            
            # Asegurar que el resultado tenga la forma de un output de Keras (batch_size, num_classes)
            return np.array([probs])

    model = MockModel()


# ==============================================================================
# --- CLASE PRINCIPAL DE LA APLICACIÓN ---
# ==============================================================================

class App(ctk.CTk):
    
    app_logo = None 
    placeholder_display = None # Nuevo atributo para el logo grande de placeholder

    def __init__(self):
        super().__init__()
        self.title("Clasificador MRI de etapas de Alzheimer")
        self.geometry("850x600") 
        self.config(bg=BG_WINDOW)
        
        self.grid_rowconfigure((0, 2), weight=0) 
        self.grid_rowconfigure(1, weight=1)       
        self.grid_columnconfigure((0, 2), weight=0)
        self.grid_columnconfigure(1, weight=1)
        
        self.image_display = None
        self.current_image = None 
        
        self.load_logo()
        self.create_widgets()
        
    def load_logo(self):
        """Carga el archivo de imagen del logo usando PIL y CTkImage, o usa placeholder.
           Crea dos versiones CTkImage: una pequeña para el título y una grande para el placeholder."""
        logo_img_data = None
        
        try:
            logo_img_data = Image.open(LOGO_PATH)
            print(f"Logo '{LOGO_PATH}' cargado exitosamente.")
            
        except FileNotFoundError:
            print(f"ADVERTENCIA: Archivo de logo '{LOGO_PATH}' no encontrado. Usando placeholder.")
            # Si falla, usamos el placeholder Base64
            logo_bytes = base64.b64decode(PLACEHOLDER_LOGO_BASE64)
            logo_img_data = Image.open(io.BytesIO(logo_bytes))
            
        except Exception as e:
            print(f"ADVERTENCIA: Error al cargar el logo: {e}. Usando placeholder.")
            logo_bytes = base64.b64decode(PLACEHOLDER_LOGO_BASE64)
            logo_img_data = Image.open(io.BytesIO(logo_bytes))
            
        finally:
            if logo_img_data:
                # 1. Logo pequeño para el título
                self.app_logo = ctk.CTkImage(
                    light_image=logo_img_data, 
                    size=(45, 45) 
                )
                # 2. Logo grande para el placeholder (marca de agua)
                self.placeholder_display = ctk.CTkImage(
                    light_image=logo_img_data, 
                    size=(150, 150) # Tamaño más grande para efecto de marca de agua
                )


    def create_widgets(self):
        
        # ----------------------------------------------------------------------
        # Marco Principal de Contenido (La "Tarjeta" Central)
        # ----------------------------------------------------------------------
        content_card_frame = ctk.CTkFrame(
            self, 
            fg_color=CARD_BG,
            corner_radius=10, 
            border_width=1,
            border_color="#DDDDDD",
        )
        content_card_frame.grid(row=1, column=1, sticky="nsew", padx=15, pady=15)
        content_card_frame.grid_columnconfigure(0, weight=1) 
        content_card_frame.grid_rowconfigure((0, 3), weight=0)
        content_card_frame.grid_rowconfigure(2, weight=1) # El frame de resultados tendrá peso
        
        # --- Bloque de Título Principal (Fila 0) ---
        title_block = ctk.CTkFrame(content_card_frame, fg_color=CARD_BG, corner_radius=0)
        title_block.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 5))
        # 📌 Corregido: Columna 0 para Logo, Columna 1 para Texto
        title_block.grid_columnconfigure(0, weight=0) # Columna 0: Logo (No se expande)
        title_block.grid_columnconfigure(1, weight=1) # Columna 1: Texto (Se expande)
                 
        # 📌 2. TÍTULO PRINCIPAL
        ctk.CTkLabel(
            title_block,
            text="Análisis MRI de Deterioro Cognitivo", 
            font=TITLE_FONT, 
            text_color=TITLE_BLUE 
        ).grid(row=0, column=1, sticky="w") 

        # 📌 3. SUBTÍTULO
        ctk.CTkLabel(
            title_block,
            text="Prediagnóstico basado en CNN", 
            font=HEADER_FONT,
            text_color=TEXT_DARK 
        ).grid(row=1, column=1, pady=(0, 5), sticky="w")
        
        ctk.CTkFrame(content_card_frame, height=1, fg_color="#E0E0E0" # Separador
        ).grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))
        
        # ----------------------------------------------------------------------
        # --- RESTO DE WIDGETS (Imagen, Resultados, Barras) ---
        # ----------------------------------------------------------------------
        
        # Marco Contenedor de Imagen y Resultados (Fila 2)
        results_and_bars_frame = ctk.CTkFrame(
            content_card_frame, 
            fg_color=CARD_BG 
        )
        results_and_bars_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        results_and_bars_frame.grid_columnconfigure(0, weight=1) 
        results_and_bars_frame.grid_columnconfigure(1, weight=1) 
        results_and_bars_frame.grid_rowconfigure(0, weight=1)

        # COLUMNA 0: ZONA DE VISUALIZACIÓN DE IMAGEN
        self.image_container = ctk.CTkFrame(
            results_and_bars_frame,
            width=380, 
            height=380, 
            fg_color="#F8F8F8", 
            border_width=1, 
            border_color="#DDDDDD",
            corner_radius=10 
        )
        self.image_container.grid(row=0, column=0, pady=10, padx=(10, 15), sticky="nsew") 
        self.image_container.pack_propagate(False) 
        
        # 📌 WIDGET PRINCIPAL DE IMAGEN/PLACEHOLDER (MODIFICADO)
        self.image_label = ctk.CTkLabel(
            self.image_container, 
            text="Esperando Imagen MRI...",
            image=self.placeholder_display, # Se agrega la imagen grande como placeholder
            compound="top",                 # Permite mostrar la imagen y el texto
            font=BODY_FONT,
            text_color="#AAAAAA" 
        )
        self.image_label.pack(expand=True)
        
        # COLUMNA 1: ZONA DE RESULTADOS Y BARRAS
        results_frame = ctk.CTkFrame(results_and_bars_frame, fg_color=CARD_BG)
        results_frame.grid(row=0, column=1, pady=10, padx=(0, 10), sticky="nsew")
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(5, weight=1) 

        # --- 1. Prediagnóstico Principal (Fila 0) ---
        ctk.CTkLabel(results_frame, text="PREDIAGNÓSTICO PRINCIPAL", font=HEADER_FONT, text_color=TITLE_BLUE
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        self.main_result_frame = ctk.CTkFrame(
            results_frame, fg_color=NEUTRAL_COLOR, corner_radius=10, height=60
        )
        self.main_result_frame.grid(row=1, column=0, pady=(0, 10), sticky="ew") 
        self.main_result_frame.grid_columnconfigure(0, weight=0)
        self.main_result_frame.grid_columnconfigure(1, weight=1)

        #  Icono del resultado principal
        self.result_icon = ctk.CTkLabel(self.main_result_frame, text="", font=ICON_FONT, text_color=TEXT_DARK)
        self.result_icon.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")
        
        self.result_label = ctk.CTkLabel(
            self.main_result_frame,
            text="Presione el botón para analizar.", 
            font=RESULT_FONT, text_color=TEXT_DARK, justify="left"
        )
        self.result_label.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="w") 
        
        # --- 2. Barra de Confianza Principal (Fila 2) ---
        ctk.CTkLabel(results_frame, text="Confianza de la Predicción:", font=BODY_FONT, text_color=TEXT_DARK
        ).grid(row=2, column=0, sticky="w", pady=(5, 5))

        self.confidence_bar = ctk.CTkProgressBar(
            results_frame, orientation="horizontal", height=15, 
            fg_color=PROGRESS_BG, progress_color=PROGRESS_FG
        )
        self.confidence_bar.set(0) 
        self.confidence_bar.grid(row=3, column=0, pady=(0, 15), sticky="ew")
        
        # --- 3. Distribución de Probabilidades (Fila 4) ---
        ctk.CTkLabel(results_frame, text="DISTRIBUCIÓN DETALLADA POR CLASE", font=HEADER_FONT, text_color=TITLE_BLUE
        ).grid(row=4, column=0, sticky="w", pady=(15, 5))

        # Marco para las barras de clases
        self.classes_frame = ctk.CTkFrame(
            results_frame, fg_color="#F8F8F8", border_width=1, border_color="#DDDDDD"
        )
        self.classes_frame.grid(row=5, column=0, sticky="nsew", pady=(0, 10))
        self.classes_frame.grid_columnconfigure(0, weight=1) 
        self.classes_frame.grid_columnconfigure(1, weight=1) 

        self.class_progress_bars = {}
        for i, class_name_raw in enumerate(MODEL_CLASS_ORDER):
            display_name = CLASS_NAMES_MAP.get(class_name_raw, class_name_raw)

            label_container = ctk.CTkFrame(self.classes_frame, fg_color="#F8F8F8")
            label_container.grid(row=i, column=0, sticky="ew", padx=(5, 0), pady=2)
            label_container.grid_columnconfigure(0, weight=1)

            label = ctk.CTkLabel(
                label_container, text=display_name, font=BODY_FONT, anchor="w", justify="left", text_color=TEXT_DARK
            )
            label.grid(row=0, column=0, sticky="w")
            
            percent_label = ctk.CTkLabel(
                label_container, text="0.00%", font=CLASS_BAR_FONT, anchor="e", text_color=TEXT_DARK
            )
            percent_label.grid(row=0, column=1, sticky="e", padx=(0, 5))

            progress_bar = ctk.CTkProgressBar(
                self.classes_frame, orientation="horizontal", height=10, 
                fg_color=PROGRESS_BG, progress_color=PROGRESS_FG
            )
            progress_bar.set(0)
            progress_bar.grid(row=i, column=1, sticky="ew", padx=(0, 5), pady=5)
            
            self.class_progress_bars[class_name_raw] = {
                'bar': progress_bar, 
                'label': label,
                'percent_label': percent_label
            }

        # Botón de Carga (Fila 3)
        self.load_button = ctk.CTkButton(
            content_card_frame,
            text="Cargar y Analizar Imagen MRI", 
            command=self.load_image,
            font=BUTTON_FONT,
            height=40, 
            fg_color=PRIMARY_BLUE, 
            hover_color=SECONDARY_BLUE,
            text_color=TEXT_LIGHT,
            corner_radius=10, 
            cursor="hand2"
        )
        self.load_button.grid(row=3, column=0, pady=(0, 15), padx=20, sticky="ew") 


    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar Imagen de Resonancia Magnética",
            filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return 
        
        try:
            # Abrir y asegurar que está en formato RGB (muchos modelos CNN esperan 3 canales)
            img = Image.open(file_path).convert("RGB") 
            self.current_image = img 
            
            # Ajuste de tamaño para la visualización
            frame_w = self.image_container.winfo_width()
            frame_h = self.image_container.winfo_height()
            
            img_w, img_h = img.size
            # Usar 90% del contenedor para tener un pequeño margen
            ratio = min(frame_w / img_w, frame_h / img_h)
            display_size = (int(img_w * ratio * 0.95), int(img_h * ratio * 0.95))

            self.image_display = ctk.CTkImage(
                light_image=img,
                size=display_size
            )
            # Centrar la imagen en el contenedor y quitar el placeholder
            self.image_label.configure(
                image=self.image_display, 
                text="",
                compound="center" # Centra la imagen y quita el texto "Esperando..."
            )
            self.image_label.pack_forget()
            self.image_label.pack(expand=True)
            
            self.predict_image(self.current_image)
            
        except Exception as e:
            messagebox.showerror("Error de Imagen", f"No se pudo procesar la imagen: {e}")
            self.update_results_display(error=True) 

    def predict_image(self, img):
        if model is None:
            self.update_results_display(error=True, message="ERROR: Modelo no disponible.")
            return

        try:
            img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = np.array(img_resized)
            
            # Preparación para el modelo: expandir dimensión de lote y normalizar si es necesario
            img_tensor = np.expand_dims(img_array, axis=0)
            
            # Es vital saber si tu modelo fue entrenado con valores normalizados (0-1)
            # Si fue entrenado con normalización, descomenta la siguiente línea:
            # img_tensor = img_tensor.astype('float32') / 255.0 

            predictions = model.predict(img_tensor, verbose=0)
            
            # Asegurar que las predicciones sean un array de probabilidades
            if predictions.ndim > 1:
                probabilities = predictions[0] * 100 
            else:
                 # Esto podría ocurrir con el mock, forzamos un formato correcto
                 probabilities = predictions * 100


            results = {}
            for i, class_name_raw in enumerate(MODEL_CLASS_ORDER):
                # Asegurarse de que el índice no esté fuera de rango
                if i < len(probabilities):
                    results[class_name_raw] = probabilities[i]
                else:
                    results[class_name_raw] = 0.0 # Valor por defecto si hay un desajuste
                    
            self.update_results_display(results=results)

            
        except Exception as e:
            messagebox.showerror("Error de Predicción", f"No se pudo realizar la predicción: {e}")
            self.update_results_display(error=True, message=f"Error en la predicción: {e}")
            print(f"ERROR en predict_image: {e}")


    def update_results_display(self, results=None, error=False, message=None):
        
        # --- Manejo de Errores / Reset a estado inicial --- (MODIFICADO)
        if error or results is None:
            # Resetear el display de imagen al placeholder
            self.image_label.configure(
                image=self.placeholder_display, # Restaura el logo grande
                text=message or "Esperando Imagen MRI...",
                compound="top", # Para que se vea el logo y el texto
                text_color="#AAAAAA"
            )
            
            # --- Manejo de Errores para Resultados ---
            self.result_label.configure(text=message or "Análisis Fallido", text_color=TEXT_LIGHT)
            self.main_result_frame.configure(fg_color=WARNING_COLOR)
            self.result_icon.configure(text="", text_color=TEXT_LIGHT)
            self.confidence_bar.set(0)
            
            for item in self.class_progress_bars.values():
                item['bar'].set(0)
                item['percent_label'].configure(text="0.00%", text_color=TEXT_DARK)
                item['label'].configure(text_color=TEXT_DARK)
            return

        # --- Obtener el resultado principal ---
        # Usamos try/except para manejar el caso de un diccionario vacío
        try:
            predicted_class_raw = max(results, key=results.get)
        except ValueError:
            self.update_results_display(error=True, message="Error: Resultados Vacíos.")
            return
            
        confidence = results[predicted_class_raw]
        predicted_class_display = CLASS_NAMES_MAP.get(predicted_class_raw, predicted_class_raw)

        # --- Actualizar Marco de Resultado Principal ---
        is_demented = predicted_class_raw != "NonDemented"
        color = WARNING_COLOR if is_demented else SUCCESS_COLOR 
        
        # Añadimos iconos para mejor feedback visual
        icon = "" if is_demented else ""
        
        result_text = f"CLASIFICACIÓN: {predicted_class_display} ({confidence:.2f}%)"
        
        self.result_label.configure(text=result_text, text_color=TEXT_LIGHT)
        self.main_result_frame.configure(fg_color=color)
        self.result_icon.configure(text=icon, text_color=TEXT_LIGHT) # Actualizar Icono
        
        
        # --- Actualizar Barra Principal de Confianza ---
        self.confidence_bar.set(confidence / 100.0)
        self.confidence_bar.configure(progress_color=color) 

        # --- Actualizar Barras de Distribución ---
        for class_name_raw, progress_data in self.class_progress_bars.items():
            prob = results.get(class_name_raw, 0.0)
            
            progress_data['bar'].set(prob / 100.0)
            
            bar_fill_color = color if class_name_raw == predicted_class_raw else PRIMARY_BLUE
            progress_data['bar'].configure(progress_color=bar_fill_color)

            # Destacar texto de la clase ganadora
            text_color = color if class_name_raw == predicted_class_raw else TEXT_DARK
            
            progress_data['percent_label'].configure(
                text=f"{prob:.2f}%",
                text_color=text_color,
                font=CLASS_BAR_FONT
            )
            progress_data['label'].configure(
                font=CLASS_BAR_FONT if class_name_raw != predicted_class_raw else RESULT_FONT, # Hacer más grande el texto de la clase principal
                text_color=text_color
            )


if __name__ == "__main__":
    app = App()
    app.mainloop()