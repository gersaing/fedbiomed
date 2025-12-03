import sys
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pydicom
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from fedbiomed.modulo_dicom.vista.clasificador import evaluar_estudio_cancer
from ...preprocesador.preprocesamiento import PipelinePreprocesamiento

from ..configuracion import obtener_rutas
from fedbiomed.modulo_dicom.pipeline_transformacion import PipelineDicom
from ..utilidades import obtener_ruta_salida_modulo
from PIL import Image, ImageTk
# Variable global para ruta actual
ruta_actual_dicom = None
ruta_img_logo = "fedbiomed/fedbiomed/modulo_dicom/vista/fedbiomed-logo.png"
def cargar_dicom():
    global ruta_actual_dicom
    ruta_archivo = filedialog.askopenfilename(
        title="Seleccionar archivo DICOM",
        filetypes=[("Archivos DICOM","*.dicom"),("Archivos DCM", "*.dcm"), ("Todos los archivos", "*.*")]
    )
    if ruta_archivo:
        ruta_actual_dicom = ruta_archivo
        try:
            dicom = pydicom.dcmread(ruta_archivo)

            # Limpiar metadatos anteriores
            texto_metadatos.delete("1.0", tk.END)

            # Mostrar metadatos (excluyendo PixelData)
            for elem in dicom:
                if elem.name != "Pixel Data":
                    texto_metadatos.insert(tk.END, f"{elem.tag} : {elem.name} = {elem.value}\n")

            # Limpiar imagen anterior
            for widget in frame_imagen.winfo_children():
                widget.destroy()

            # Mostrar imagen si existe
            if 'PixelData' in dicom:
                # --- Aumentamos el tamaño de la figura y del lienzo ---
                fig, ax = plt.subplots(figsize=(8, 8))  # antes 5x5
                ax.imshow(dicom.pixel_array, cmap='gray')
                ax.axis('off')

                # Creamos el canvas dentro del frame, más grande y expansible
                canvas = FigureCanvasTkAgg(fig, master=frame_imagen)
                canvas.draw()
                widget_canvas = canvas.get_tk_widget()
                widget_canvas.pack(fill=tk.BOTH, expand=True)
                widget_canvas.config(width=700, height=700)  # aumenta el área visible

                # ========== ZOOM Y PAN ==========
                def on_scroll(event):
                    """Zoom con la rueda del ratón"""
                    base_scale = 1.2
                    if event.button == 'up':
                        scale_factor = 1 / base_scale
                    elif event.button == 'down':
                        scale_factor = base_scale
                    else:
                        return

                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    xdata = event.xdata
                    ydata = event.ydata

                    if xdata is None or ydata is None:
                        return  # Evita errores si el mouse está fuera del eje

                    ax.set_xlim([xdata - (xdata - xlim[0]) * scale_factor,
                                 xdata + (xlim[1] - xdata) * scale_factor])
                    ax.set_ylim([ydata - (ydata - ylim[0]) * scale_factor,
                                 ydata + (ylim[1] - ydata) * scale_factor])
                    canvas.draw_idle()

                press_event = {"x": None, "y": None}

                def on_press(event):
                    if event.button == 1:
                        press_event["x"], press_event["y"] = event.x, event.y

                def on_motion(event):
                    if press_event["x"] is None or press_event["y"] is None:
                        return
                    dx = event.x - press_event["x"]
                    dy = event.y - press_event["y"]
                    press_event["x"], press_event["y"] = event.x, event.y

                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    ax.set_xlim(xlim[0] - dx / 10, xlim[1] - dx / 10)
                    ax.set_ylim(ylim[0] + dy / 10, ylim[1] + dy / 10)
                    canvas.draw_idle()

                def on_release(event):
                    press_event["x"], press_event["y"] = None, None

                canvas.mpl_connect("scroll_event", on_scroll)
                canvas.mpl_connect("button_press_event", on_press)
                canvas.mpl_connect("motion_notify_event", on_motion)
                canvas.mpl_connect("button_release_event", on_release)
                # ===================================

                messagebox.showinfo("Carga", "DICOM cargado correctamente.")
                boton_procesar.config(state=tk.NORMAL)

            else:
                messagebox.showinfo("Sin imagen", "El archivo DICOM no contiene datos de imagen.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo DICOM:\n{e}")



def procesar_dicom():
    if ruta_actual_dicom is None:
        messagebox.showwarning("Sin archivo", "Primero debes cargar un archivo DICOM.")
        return
    try:
        print("[INFO] Iniciando procesamiento de archivos DICOM...")
        ruta_nodo = obtener_ruta_salida_modulo("fbm-soli-node") #Temporal
        rutas = obtener_rutas(ruta_nodo)
        ruta_dicoms = ruta_actual_dicom
        limite_dicoms = 1  # Ajusta según tu lógica

        pipeline = PipelineDicom(rutas, ruta_dicoms, limite_dicoms)
        print("[INFO] ruta_dicoms:", ruta_dicoms)
        pipeline.procesar_directorio()
        pipeline.extraer_caracteristicas()
        pipeline.fusionar_caracteristicas_metadatos()

        # 3) Preprocesamiento (normalizacion, codificacion)
        pipelinePreprocesamiento = PipelinePreprocesamiento()
        df = pd.read_csv(rutas["fusionado"])
        df = pipelinePreprocesamiento.eliminar_columnas_irrelevantes(df)
        df = pipelinePreprocesamiento.codificar_dataset(df)
        df = pipelinePreprocesamiento.ajustar_dataset_validacion(df, rutas["entrenamiento"])
        df = pipelinePreprocesamiento.normalizar_validacion(df, rutas["escalador"],rutas["validacion"])
        #el flujo según si es entrenamiento o validación

        messagebox.showinfo("Procesamiento", "Procesamiento del DICOM completado.")
        boton_evaluar.config(state=tk.NORMAL)

    except Exception as e:
        messagebox.showerror("Error", f"No se pudo procesar el archivo DICOM:\n{e}")


def cerrar_ventana():
    ventana.destroy()
    sys.exit()

import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
from PIL import Image, ImageTk

# ========== CONFIGURACIÓN GENERAL ==========
COLOR_FONDO = "#f5f5f5"
COLOR_BOTON = "#003366"
COLOR_BOTON_ACTIVO = "#003366"
FUENTE_BOTON = ("Segoe UI", 10, "bold")

# ========== VENTANA PRINCIPAL ==========
ventana = tk.Tk()
ventana.title("Visualizador DICOM")
ventana.geometry("1000x600")
ventana.configure(bg=COLOR_FONDO)

# ========== LOGO SUPERIOR IZQUIERDO ==========
frame_logo = tk.Frame(ventana, bg=COLOR_FONDO)
frame_logo.pack(side=tk.TOP, anchor='nw', padx=10, pady=10)

try:
    logo_img = Image.open("fedbiomed/fedbiomed/modulo_dicom/vista/fedbiomed-logo.png")  # Ajusta la ruta si es necesario
    logo_img = logo_img.resize((60, 60), Image.Resampling.LANCZOS)
    logo_tk = ImageTk.PhotoImage(logo_img)
    logo_label = tk.Label(frame_logo, image=logo_tk, bg=COLOR_FONDO)
    logo_label.image = logo_tk
    logo_label.pack()
except Exception as e:
    logo_label = tk.Label(frame_logo, text="LOGO", bg=COLOR_FONDO, font=("Segoe UI", 12, "bold"))
    logo_label.pack()

# ========== BOTONES PRINCIPALES ==========
frame_botones = tk.Frame(ventana, bg=COLOR_FONDO)
frame_botones.pack(pady=10)

estilo_boton = {
    "bg": COLOR_BOTON,
    "fg": "white",
    "font": FUENTE_BOTON,
    "activebackground": COLOR_BOTON_ACTIVO,
    "activeforeground": "white",
    "bd": 0,
    "relief": tk.FLAT,
    "padx": 10,
    "pady": 5
}
def evaluar_estudio():
    boton_evaluar.config(state='disabled')
    boton_procesar.config(state='disabled')
    ventana.update_idletasks()  # Fuerza actualización visual inmediata
    evaluar_estudio_cancer()


boton_cargar = tk.Button(frame_botones, text="Cargar DICOM", command=cargar_dicom, **estilo_boton)
boton_cargar.pack(side=tk.LEFT, padx=5)

boton_procesar = tk.Button(frame_botones, text="Procesar DICOM", command=procesar_dicom, **estilo_boton)
boton_procesar.pack(side=tk.LEFT, padx=5)
boton_procesar.config(state=tk.DISABLED)


boton_evaluar = tk.Button(frame_botones, text="Evaluar estudio", command=evaluar_estudio, **estilo_boton)
boton_evaluar.pack(side=tk.LEFT, padx=5)
boton_evaluar.config(state=tk.DISABLED)



# ========== PANEL DE IMAGEN ==========
frame_imagen = tk.Frame(ventana, bg=COLOR_FONDO, width=700, height=700)
frame_imagen.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)


# ========== METADATOS CON SCROLL ==========
texto_metadatos = scrolledtext.ScrolledText(ventana, width=60, bg="white", fg="black", font=("Consolas", 10))
texto_metadatos.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

# ========== INICIO DE INTERFAZ ==========
def iniciar_interfaz():
    ventana.protocol("WM_DELETE_WINDOW", cerrar_ventana)
    ventana.mainloop()