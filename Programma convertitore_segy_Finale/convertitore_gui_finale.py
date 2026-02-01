#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from io import StringIO
from obspy import Trace, Stream, UTCDateTime

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# LETTURA DATI (virgole → punti)
# ------------------------------------------------------------
def read_data(input_path):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw = f.read()

        raw = raw.replace(",", ".")
        data = np.loadtxt(StringIO(raw))

    except Exception as e:
        raise RuntimeError(f"Errore nella lettura del file: {e}")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data


# ------------------------------------------------------------
# GEOMETRIA
# ------------------------------------------------------------
def build_geometry_metadata(n_channels, dx, x0, source_offset):
    receiver_positions = [x0 + i * dx for i in range(n_channels)]
    source_position = x0 + source_offset if source_offset is not None else None
    return receiver_positions, source_position


# ------------------------------------------------------------
# CONVERSIONE IN SEGY
# ------------------------------------------------------------
def convert_to_segy(input_path, sampling_rate, method, spread_type,
                    dx, x0, source_offset, station, network, starttime, output_path=None):

    if not os.path.exists(input_path):
        raise RuntimeError(f"Il file {input_path} non esiste.")

    if not output_path:
        output_path = os.path.splitext(input_path)[0] + ".sgy"

# Start time fittizio per far partire l'asse X da 0
    start_time = UTCDateTime("2000-01-01T00:00:00")

    data = read_data(input_path)
    n_samples, n_channels = data.shape

    receiver_positions, source_position = build_geometry_metadata(
        n_channels, dx, x0, source_offset
    )

    st = Stream()

    for i in range(n_channels):
        chan_id = i + 1

        stats = {
            "network": network,
            "station": station,
            "location": "",
            "channel": f"{chan_id}",
            "npts": n_samples,
            "sampling_rate": sampling_rate,
            "starttime": start_time,
        }

        trace_data = data[:, i].astype(np.float32)
        tr = Trace(data=trace_data, header=stats)

        tr.stats.segy = {"trace_header": {}}
        th = tr.stats.segy.trace_header

        x_rec = receiver_positions[i]
        th.group_coordinate_x = int(round(x_rec))
        th.group_coordinate_y = 0
        th.group_coordinate_z = 0

        if source_position is not None:
            th.source_coordinate_x = int(round(source_position))
            th.source_coordinate_y = 0
            th.source_coordinate_z = 0

        th.trace_sequence_number_within_line = chan_id
        th.energy_source_point_number = 1

        st.append(tr)

    try:
        st.write(output_path, format="SEGY")
    except Exception as e:
        raise RuntimeError(f"Errore scrittura SEGY: {e}")

    return output_path


# ------------------------------------------------------------
# GUI
# ------------------------------------------------------------
class ConverterGUI:
    def __init__(self, master):
        self.master = master
        master.title("Convertitore Sismico TXT/CSV → SEGY")

        self.input_file = tk.StringVar()
        self.sampling_rate = tk.StringVar(value="500")
        self.method = tk.StringVar(value="MASW")
        self.spread_type = tk.StringVar(value="lineare")
        self.dx = tk.StringVar(value="1.0")
        self.x0 = tk.StringVar(value="0.0")
        self.source_offset = tk.StringVar(value="-2.0")
        self.station = tk.StringVar(value="MASW")
        self.network = tk.StringVar(value="XX")
        self.starttime = tk.StringVar(value="")
        self.output_file = tk.StringVar(value="")

        row = 0

        tk.Label(master, text="File input (TXT/CSV):").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(master, textvariable=self.input_file, width=50).grid(row=row, column=1, padx=5, pady=5)
        tk.Button(master, text="Sfoglia...", command=self.browse_file).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        tk.Label(master, text="Frequenza campionamento (Hz):").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(master, textvariable=self.sampling_rate, width=10).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1

        tk.Label(master, text="Metodo:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        ttk.Combobox(master, textvariable=self.method,
                     values=["MASW", "ReMi", "Rifrazione", "Altro"],
                     width=10, state="readonly").grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1

        tk.Label(master, text="Tipo stendimento:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        ttk.Combobox(master, textvariable=self.spread_type,
                     values=["lineare", "end-on", "split-spread", "altro"],
                     width=10, state="readonly").grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1

        tk.Label(master, text="dx geofoni (m):").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(master, textvariable=self.dx, width=10).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1

        tk.Label(master, text="x0 primo geofono (m):").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(master, textvariable=self.x0, width=10).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1

        tk.Label(master, text="Offset sorgente (m):").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(master, textvariable=self.source_offset, width=10).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1

        tk.Label(master, text="Stazione:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(master, textvariable=self.station, width=10).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1

        tk.Label(master, text="Network:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(master, textvariable=self.network, width=10).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1

        tk.Label(master, text="Start time (ISO, opzionale):").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(master, textvariable=self.starttime, width=20).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1

        tk.Label(master, text="File output (.sgy):").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(master, textvariable=self.output_file, width=50).grid(row=row, column=1, padx=5, pady=5)
        tk.Button(master, text="Sfoglia...", command=self.browse_output).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        tk.Button(master, text="Mostra anteprima tracce", command=self.show_preview).grid(
            row=row, column=0, pady=10
        )
        tk.Button(master, text="Converti in SEGY", command=self.run_conversion,
                  bg="#4CAF50", fg="white").grid(row=row, column=1, columnspan=2, pady=10, sticky="w")
        row += 1


    # ------------------------------------------------------------
    # FUNZIONI FILE DIALOG
    # ------------------------------------------------------------
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Seleziona file TXT/CSV",
            filetypes=[("Testo/CSV", "*.txt *.csv *.dat"), ("Tutti i file", "*.*")]
        )
        if filename:
            self.input_file.set(filename)

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Salva come SEGY",
            defaultextension=".sgy",
            filetypes=[("SEGY", "*.sgy"), ("Tutti i file", "*.*")]
        )
        if filename:
            self.output_file.set(filename)


    # ------------------------------------------------------------
    # ANTEPRIMA COMPLETA
    # ------------------------------------------------------------
    def show_preview(self):
        try:
            if not self.input_file.get():
                messagebox.showerror("Errore", "Seleziona un file di input prima di mostrare l'anteprima.")
                return

            data = read_data(self.input_file.get())

            try:
                fs = float(self.sampling_rate.get())
            except ValueError:
                fs = 1.0

            n_samples, n_channels = data.shape
            t = np.arange(n_samples) / fs

            win = tk.Toplevel(self.master)
            win.title("Anteprima tracce")

            # Pulsanti
            btn_frame = tk.Frame(win)
            btn_frame.pack(pady=5)

            autoscale_btn = tk.Button(btn_frame, text="Autoscale")
            autoscale_btn.grid(row=0, column=0, padx=5)

            reset_btn = tk.Button(btn_frame, text="Reset Zoom")
            reset_btn.grid(row=0, column=1, padx=5)

            save_btn = tk.Button(btn_frame, text="Salva PNG")
            save_btn.grid(row=0, column=2, padx=5)

            overlay_btn = tk.Button(btn_frame, text="Tracce Sovrapposte")
            overlay_btn.grid(row=0, column=3, padx=5)
            stacked_btn = tk.Button(btn_frame, text="Tracce traslate")
            stacked_btn.grid(row=0, column=4, padx=5)


            # Griglia automatica
            if n_channels <= 4:
                rows, cols = 2, 2
            elif n_channels <= 6:
                rows, cols = 2, 3
            elif n_channels <= 9:
                rows, cols = 3, 3
            elif n_channels <= 12:
                rows, cols = 3, 4
            else:
                rows = int(np.ceil(np.sqrt(n_channels)))
                cols = rows

            fig = Figure(figsize=(10, 7), dpi=100)

            cmap = plt.get_cmap("tab20")
            colors = [cmap(i % 20) for i in range(n_channels)]

            axes = []
            original_limits = []

            # Disegna pannelli
            for i in range(n_channels):
                ax = fig.add_subplot(rows, cols, i + 1)
                axes.append(ax)

                trace = data[:, i]
                scale = 1.0 / np.max(np.abs(trace)) if np.max(np.abs(trace)) != 0 else 1.0
                trace_scaled = trace * scale

                ax.plot(t, trace_scaled, color=colors[i], linewidth=0.8)
                ax.set_title(f"Ch {i+1}", fontsize=8)
                ax.grid(True)
                ax.set_xticks([])
                ax.set_yticks([])

                original_limits.append(ax.get_ylim())

            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # AUTOSCALE
            def autoscale():
                for i, ax in enumerate(axes):
                    trace = data[:, i]
                    ymin, ymax = np.min(trace), np.max(trace)
                    if ymin == ymax:
                        ymin -= 1
                        ymax += 1
                    ax.set_ylim(ymin, ymax)
                canvas.draw()

            autoscale_btn.config(command=autoscale)

            # RESET ZOOM
            def reset_zoom():
                for ax, lim in zip(axes, original_limits):
                    ax.set_ylim(lim)
                canvas.draw()

            reset_btn.config(command=reset_zoom)

            # SALVA PNG
            def save_png():
                filename = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG Image", "*.png")]
                )
                if filename:
                    fig.savefig(filename, dpi=150)
                    messagebox.showinfo("Salvato", f"Immagine salvata:\n{filename}")

            save_btn.config(command=save_png)

            # TRACCE SOVRAPPPOSTE
            def overlay():
                overlay_win = tk.Toplevel(win)
                overlay_win.title("Tracce Sovrapposte")

                fig2 = Figure(figsize=(10, 6), dpi=100)
                ax2 = fig2.add_subplot(111)

                for i in range(n_channels):
                    trace = data[:, i]
                    scale = 1.0 / np.max(np.abs(trace)) if np.max(np.abs(trace)) != 0 else 1.0
                    ax2.plot(t, trace * scale, linewidth=0.8, color=colors[i], label=f"Ch {i+1}")

                ax2.set_xlabel("Tempo (s)")
                ax2.set_ylabel("Ampiezza (scalata)")
                ax2.grid(True)
                ax2.legend(fontsize=6, ncol=4)

                canvas2 = FigureCanvasTkAgg(fig2, master=overlay_win)
                canvas2.draw()
                canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            def stacked():
                stacked_win = tk.Toplevel(win)
                stacked_win.title("Tracce traslate (Ch1 in alto)")

                fig3 = Figure(figsize=(10, 6), dpi=100)
                ax3 = fig3.add_subplot(111)

                # scala globale per tutte le tracce
                max_abs = np.max(np.abs(data))
                scale = 1.0 / max_abs if max_abs != 0 else 1.0

                offset_step = 2.0
                # Ch1 in alto, poi verso il basso
                start_offset = (n_channels - 1) * offset_step

                for i in range(n_channels):
                    idx = i  # geofono i → Ch i+1
                    trace = data[:, idx] * scale
                    offset = start_offset - i * offset_step
                    ax3.plot(t, trace + offset, linewidth=0.8, color=colors[idx])
                    ax3.text(t[0], offset, f"Ch {idx+1}", va="bottom", fontsize=7)

                ax3.set_xlabel("Tempo (s)")
                ax3.set_ylabel("Tracce (scalate e traslate)")
                ax3.grid(True)

                fig3.tight_layout()
                canvas3 = FigureCanvasTkAgg(fig3, master=stacked_win)
                canvas3.draw()
                canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            stacked_btn.config(command=stacked)

            overlay_btn.config(command=overlay)

        except Exception as e:
            messagebox.showerror("Errore anteprima", str(e))


    # ------------------------------------------------------------
    # CONVERSIONE
    # ------------------------------------------------------------
    def run_conversion(self):
        try:
            if not self.input_file.get():
                messagebox.showerror("Errore", "Seleziona un file di input.")
                return

            fs = float(self.sampling_rate.get())
            dx_val = float(self.dx.get())
            x0_val = float(self.x0.get())
            so_val = float(self.source_offset.get())

            out = self.output_file.get().strip()
            if out == "":
                out = None

            output_path = convert_to_segy(
                input_path=self.input_file.get(),
                sampling_rate=fs,
                method=self.method.get(),
                spread_type=self.spread_type.get(),
                dx=dx_val,
                x0=x0_val,
                source_offset=so_val,
                station=self.station.get(),
                network=self.network.get(),
                starttime=self.starttime.get().strip(),
                output_path=out
            )

            messagebox.showinfo("Successo", f"File SEGY creato:\n{output_path}")

        except Exception as e:
            messagebox.showerror("Errore", str(e))


# ------------------------------------------------------------
# AVVIO APP
# ------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ConverterGUI(root)
    root.mainloop()
