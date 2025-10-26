import os
import math
import tempfile
import wave
import struct
import numpy as np
import random
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# Попытка импортировать pydub — не критично
try:
    from pydub import AudioSegment
    _HAS_PYDUB = True
except Exception:
    _HAS_PYDUB = False

image_data = None
canvas_img_refs = []
current_width = None
current_height = None

def setup_clipboard_bindings(widget):
    def gen(event_name):
        return lambda e: (widget.event_generate(event_name), "break")

    widget.bind("<Control-c>", gen("<<Copy>>"))
    widget.bind("<Control-v>", gen("<<Paste>>"))
    widget.bind("<Control-x>", gen("<<Cut>>"))
    widget.bind("<Control-a>", lambda e: (widget.tag_add("sel", "1.0", "end"), "break"))

    widget.bind("<Command-c>", gen("<<Copy>>"))
    widget.bind("<Command-v>", gen("<<Paste>>"))
    widget.bind("<Command-x>", gen("<<Cut>>"))
    widget.bind("<Command-a>", lambda e: (widget.tag_add("sel", "1.0", "end"), "break"))

    widget.bind("<Button-1>", lambda e: widget.focus_set())

    menu = tk.Menu(widget, tearoff=0)
    menu.add_command(label="Копировать", command=lambda: widget.event_generate("<<Copy>>"))
    menu.add_command(label="Вставить", command=lambda: widget.event_generate("<<Paste>>"))
    menu.add_command(label="Вырезать", command=lambda: widget.event_generate("<<Cut>>"))
    menu.add_separator()
    menu.add_command(label="Выделить всё", command=lambda: widget.tag_add("sel", "1.0", "end"))

    def show_menu(event):
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    widget.bind("<Button-3>", show_menu)
    widget.bind("<Control-Button-1>", show_menu)  # для macOS

def load_image():
    global image_data, current_width, current_height
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files","*.*")])
    if not path:
        return
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось открыть изображение: {e}")
        return

    image_data = np.array(img)
    current_height, current_width = image_data.shape[:2]
    width_var.set(str(current_width))

    win = tk.Toplevel(root)
    win.title(f"Изображение — {os.path.basename(path)}")
    canvas = tk.Canvas(win, width=img.width, height=img.height)
    canvas.pack()
    photo = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas_img_refs.append(photo)

    fill_text_from_image(image_data)

def fill_text_from_image(arr):
    h, w = arr.shape[:2]
    # Предупреждение для очень больших изображений
    max_cells_warn = 500000
    total = h * w
    if total > max_cells_warn:
        if not messagebox.askyesno("Большое изображение", f"Изображение содержит {total} пикселей. Это создаст {total} строк в табло и может сильно замедлить интерфейс. Продолжить?"):
            return

    lines = []
    # Идём в порядке строк (row-major)
    for row in arr:
        for px in row:
            r, g, b = int(px[0]), int(px[1]), int(px[2])
            lines.append(f"{r} {g} {b}")
    text_widget.config(state="normal")
    text_widget.delete("1.0", tk.END)
    text_widget.insert("1.0", "\n".join(lines))

def parse_rgb_text(text):
    pixels = []
    for i, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "," in line:
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        else:
            parts = [p for p in line.split() if p != ""]
        if len(parts) != 3:
            raise ValueError(f"Строка {i}: ожидается 3 числа (R G B), найдено {len(parts)}: '{raw_line}'")
        try:
            r, g, b = [int(p) for p in parts]
        except:
            raise ValueError(f"Строка {i}: неверный формат чисел: '{raw_line}'")
        for v in (r, g, b):
            if v < 0 or v > 255:
                raise ValueError(f"Строка {i}: значение {v} вне диапазона 0-255")
        pixels.append([r, g, b])
    if not pixels:
        raise ValueError("Не найдено ни одного RGB-триплета.")
    return pixels

def parse_numeric_grid(text):
    """Парсит текст как таблицу чисел: каждая строка — ряд чисел, разделённых пробелами или запятыми.
    Возвращает list[list[int]]; если строки имеют разную длину — бросает ValueError."""
    rows = []
    for i, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "," in line and not line.replace(',', '').replace('.', '').replace('-', '').replace(' ', '').isdigit():
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        else:
            parts = [p for p in line.split() if p != ""]
        if not parts:
            continue
        try:
            nums = [int(float(p)) for p in parts]
        except Exception:
            raise ValueError(f"Строка {i}: неверный формат числа: '{raw_line}'")
        rows.append(nums)
    if not rows:
        raise ValueError("Не найдено ни одной числовой строки.")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"Непостоянная ширина строк: найдено разные количества колонок: {sorted(list(widths))}")
    return rows


def open_image_from_text():
    global image_data, current_width, current_height
    txt = text_widget.get("1.0", tk.END)
    try:
        pixels = parse_rgb_text(txt)
    except ValueError as e:
        messagebox.showerror("Ошибка парсинга", str(e))
        return

    w_text = width_var.get().strip()
    if w_text:
        try:
            w = int(w_text)
            if w <= 0:
                raise ValueError()
        except:
            messagebox.showerror("Ошибка", "Поле ширины должно содержать положительное целое число.")
            return
    else:
        n = len(pixels)
        sq = int(np.sqrt(n))
        if sq * sq == n:
            w = sq
        else:
            messagebox.showinfo("Уточнение", "Ширина не указана и длина не является квадратом. Пожалуйста, укажите ширину.")
            return

    if len(pixels) % w != 0:
        messagebox.showerror("Ошибка", f"Количество пикселей ({len(pixels)}) не делится на указанную ширину ({w}).")
        return

    arr = np.array(pixels, dtype=np.uint8)
    h = arr.shape[0] // w
    arr = arr.reshape((h, w, 3))
    image_data = arr
    current_height, current_width = h, w
    width_var.set(str(w))

    img = Image.fromarray(arr)
    win = tk.Toplevel(root)
    win.title("Изображение из RGB")
    canvas = tk.Canvas(win, width=img.width, height=img.height)
    canvas.pack()
    photo = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas_img_refs.append(photo)

def clear_text():
    text_widget.config(state="normal")
    text_widget.delete("1.0", tk.END)

def get_desktop_path():
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "Desktop"),
        os.path.join(home, "Рабочий стол"),
        os.path.join(home, "Рабочий_стол")
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    # fallback
    return home

def image_to_audio(arr,
                   total_duration=60.0,
                   sample_rate=22050,
                   f_min=200.0,
                   f_max=6000.0,
                   max_width=300,
                   max_height=128):
    """
    Преобразует RGB-массив (H,W,3) в моно аудиосигнал numpy float32 в диапазоне [-1,1].
    - Общая длительность аудио = total_duration (сек).
    - Каждая колонка изображения занимает duration_per_column = total_duration / width.
    - Изображение может быть уменьшено для скорости.
    """
    if total_duration <= 0:
        raise ValueError("total_duration должен быть положительным числом.")

    # Приведём к RGB numpy
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    h, w = arr.shape[:2]

    # Downscale если слишком большое
    scale_w = min(1.0, max_width / max(1, w))
    scale_h = min(1.0, max_height / max(1, h))
    scale = min(scale_w, scale_h)
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR)
        arr = np.array(img)
        h, w = arr.shape[:2]

    # Конвертируем в яркость для простоты
    brightness = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0

    # Вычисляем длительность сегмента на колонку
    duration_per_column = total_duration / max(1, w)
    # Минимальная длина кадра в сэмплах — 1
    frame_len = max(1, int(sample_rate * duration_per_column))
    total_len = frame_len * w
    out = np.zeros(total_len, dtype=np.float32)
    t_frame = np.linspace(0, frame_len / sample_rate, frame_len, endpoint=False)

    # Предварительный массив частот для каждой строки (строка 0 -> верх -> высокая частота)
    if h == 1:
        freqs = np.array([(f_min + f_max) / 2.0])
    else:
        freqs = f_min + (f_max - f_min) * (1.0 - np.arange(h) / (h - 1))

    for col in range(w):
        col_b = brightness[:, col]
        active_idx = np.where(col_b > 0.001)[0]
        if active_idx.size == 0:
            seg = np.zeros(frame_len, dtype=np.float32)
        else:
            seg = np.zeros(frame_len, dtype=np.float32)
            for r in active_idx:
                amp = float(col_b[r])
                if amp <= 0:
                    continue
                f = freqs[r]
                seg += amp * np.sin(2.0 * np.pi * f * t_frame)
            # Нормализация сегмента
            max_abs = np.max(np.abs(seg))
            if max_abs > 0:
                seg /= max_abs
            seg *= min(1.0, 0.9 * (np.mean(col_b) * h / 8.0 + 0.1))
        out[col * frame_len:(col + 1) * frame_len] = seg

    # Нормализация на [-1,1]
    maxv = np.max(np.abs(out))
    if maxv > 0:
        out = out / maxv * 0.95
    return out, sample_rate

def save_wav_from_array(samples, sr, path_wav):
    """Сохранение wav 16-bit моно."""
    int_samples = np.int16(samples * 32767)
    with wave.open(path_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bit
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())

def get_array_from_widget_or_loaded():
    """Возвращает numpy-массив RGB (H,W,3) либо из image_data, либо собранный из текста. В случае ошибки показывает messagebox и возвращает None."""
    global image_data
    if image_data is not None:
        return image_data.copy()

    txt = text_widget.get("1.0", tk.END)
    try:
        pixels = parse_rgb_text(txt)
    except ValueError as e:
        messagebox.showerror("Ошибка парсинга", str(e))
        return None

    w_text = width_var.get().strip()
    if w_text:
        try:
            w = int(w_text)
            if w <= 0:
                raise ValueError()
        except:
            messagebox.showerror("Ошибка", "Поле ширины должно содержать положительное целое число.")
            return None
    else:
        n = len(pixels)
        sq = int(np.sqrt(n))
        if sq * sq == n:
            w = sq
        else:
            messagebox.showinfo("Уточнение", "Ширина не указана и длина не является квадратом. Пожалуйста, укажите ширину.")
            return None

    if len(pixels) % w != 0:
        messagebox.showerror("Ошибка", f"Количество пикселей ({len(pixels)}) не делится на указанную ширину ({w}).")
        return None

    arr = np.array(pixels, dtype=np.uint8)
    h = arr.shape[0] // w
    arr = arr.reshape((h, w, 3))
    return arr

def create_random_line():
    """
    Создаёт случайную «линию»: для каждой строки изображения выбирается случайный столбец.
    Результат вставляется в табло (по одному триплету на строку, ширина = 1)
    и открывается окно с увеличенным визуальным представлением выбранной линии.
    """
    arr = get_array_from_widget_or_loaded()
    if arr is None:
        return

    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        messagebox.showerror("Ошибка", "Неверные размеры изображения.")
        return

    # Для каждой строки выбираем случайную колонку 0..w-1
    cols = np.random.randint(0, w, size=h)
    selected = arr[np.arange(h), cols]  # shape (h,3)

    # Записываем выбранные пиксели в табло (ширина = 1)
    lines = [f"{int(r)} {int(g)} {int(b)}" for (r, g, b) in selected]
    text_widget.config(state="normal")
    text_widget.delete("1.0", tk.END)
    text_widget.insert("1.0", "\n".join(lines))
    width_var.set("1")

    # Создаём увеличенное визуальное представление (повторяем колонку по ширине для удобного просмотра)
    display_w = min(200, max(10, int(w / 2)))
    vis = np.tile(selected[:, None, :], (1, display_w, 1)).astype(np.uint8)
    img = Image.fromarray(vis)

    win = tk.Toplevel(root)
    win.title("Случайная линия (увеличенная)")
    canvas = tk.Canvas(win, width=img.width, height=img.height)
    canvas.pack()
    photo = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas_img_refs.append(photo)

    messagebox.showinfo("Готово", "Случайная линия создана и вставлена в табло (ширина = 1). Окно с визуализацией открыто.")

def create_zigzag_numbers():
    """
    Берёт табличку чисел (каждая строка — ряд чисел, разделённых пробелами или запятыми),
    проверяет, что количество колонок в каждой строке одинаково, и формирует новую колонку,
    выбирая для строки i элемент в колонке согласно зигзагообразному (ping-pong) обходу по ширине.
    Результат (по одному числу на строку) вставляется в табло, и открывается визуализация.

    Пример: для ширины 3 последовательность колонок: 0,1,2,1,0,1,2,...
    """
    txt = text_widget.get("1.0", tk.END)
    try:
        grid = parse_numeric_grid(txt)
    except ValueError as e:
        messagebox.showerror("Ошибка парсинга", str(e))
        return

    h = len(grid)
    w = len(grid[0])
    if w <= 0 or h <= 0:
        messagebox.showerror("Ошибка", "Пустая таблица.")
        return

    cols = []
    if w == 1:
        cols = [0] * h
    else:
        period = 2 * (w - 1)
        for i in range(h):
            pos = i % period
            col = pos if pos < w else period - pos
            cols.append(col)

    selected = [grid[i][cols[i]] for i in range(h)]

    # Вставляем результат в табло — одна колонка
    lines = [str(int(v)) for v in selected]
    text_widget.config(state="normal")
    text_widget.delete("1.0", tk.END)
    text_widget.insert("1.0", "\n".join(lines))
    width_var.set("1")

    # Визуализация: создаём вертикальную колонку с повторением для ширины
    try:
        vals = np.clip(np.array(selected, dtype=np.int32), 0, 255).astype(np.uint8)
    except Exception:
        vals = np.array([0]*h, dtype=np.uint8)
    display_w = min(200, max(10, int(w * 10)))
    vis = np.tile(vals[:, None], (1, display_w))  # shape (h, display_w)
    img = Image.fromarray(vis, mode='L')

    win = tk.Toplevel(root)
    win.title("Зигзаг — выбранные числа (увеличено)")
    canvas = tk.Canvas(win, width=img.width, height=img.height)
    canvas.pack()
    photo = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas_img_refs.append(photo)

    messagebox.showinfo("Готово", "Зигзаг-колонка создана и вставлена в табло (ширина = 1). Окно с визуализацией открыто.")


def save_audio():
    """Берёт либо загруженное изображение, либо строит его из текста, генерирует звук и сохраняет mp3 на рабочем стол."""
    global image_data
    arr = None
    if image_data is not None:
        arr = image_data
    else:
        # Попробуем собрать изображение из текста
        txt = text_widget.get("1.0", tk.END)
        try:
            pixels = parse_rgb_text(txt)
        except ValueError as e:
            messagebox.showerror("Ошибка парсинга", str(e))
            return
        w_text = width_var.get().strip()
        if w_text:
            try:
                w = int(w_text)
                if w <= 0:
                    raise ValueError()
            except:
                messagebox.showerror("Ошибка", "Поле ширины должно содержать положительное целое число.")
                return
        else:
            n = len(pixels)
            sq = int(np.sqrt(n))
            if sq * sq == n:
                w = sq
            else:
                messagebox.showinfo("Уточнение", "Ширина не указана и длина не является квадратом. Пожалуйста, укажите ширину.")
                return
        if len(pixels) % w != 0:
            messagebox.showerror("Ошибка", f"Количество пикселей ({len(pixels)}) не делится на указанную ширину ({w}).")
            return
        arr = np.array(pixels, dtype=np.uint8)
        h = arr.shape[0] // w
        arr = arr.reshape((h, w, 3))

    # Получаем длительность из поля
    dur_text = duration_var.get().strip()
    if not dur_text:
        total_duration = 60.0
    else:
        try:
            total_duration = float(dur_text)
            if total_duration <= 0:
                raise ValueError()
        except:
            messagebox.showerror("Ошибка", "Поле 'Длительность' должно содержать положительное число (секунды).")
            return

    # Предупреждение при слишком долгом аудио
    if total_duration > 600:
        if not messagebox.askyesno("Длинное аудио", f"Вы задали длительность {total_duration} секунд (>600). Генерация и экспорт могут занять много времени. Продолжить?"):
            return

    # Генерируем аудио
    try:
        samples, sr = image_to_audio(arr, total_duration=total_duration)
    except Exception as e:
        messagebox.showerror("Ошибка генерации аудио", f"Ошибка при генерации звука: {e}")
        return

    desktop = get_desktop_path()
    base_name = "sonification"
    mp3_path = os.path.join(desktop, f"{base_name}.mp3")
    wav_temp = None
    try:
        # Сначала временный WAV
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        wav_temp = tmp_path
        save_wav_from_array(samples, sr, wav_temp)

        if _HAS_PYDUB:
            try:
                audio = AudioSegment.from_wav(wav_temp)
                audio.export(mp3_path, format="mp3")
                messagebox.showinfo("Готово", f"MP3 сохранен на рабочем столе: {mp3_path}")
            except Exception as e:
                # возможно ffmpeg не найден
                fallback = os.path.join(desktop, "sonification.wav")
                os.replace(wav_temp, fallback)
                wav_temp = None
                messagebox.showwarning("Внимание",
                                       f"Не удалось экспортировать MP3 через pydub/ffmpeg: {e}\n"
                                       f"Сохранён WAV: {fallback}")
        else:
            # Без pydub — сохраняем WAV и предупреждаем
            fallback = os.path.join(desktop, "sonification.wav")
            os.replace(wav_temp, fallback)
            wav_temp = None
            messagebox.showinfo("Сохранено", f"pydub не установлен или ffmpeg недоступен. WAV сохранён: {fallback}\n"
                                            f"Чтобы получить MP3, установите pydub и ffmpeg.")
    finally:
        if wav_temp and os.path.exists(wav_temp):
            os.remove(wav_temp)

root = tk.Tk()
root.title("RGB редактор Tkinter — Sonification")
root.geometry("980x760")

top_frame = tk.Frame(root)
top_frame.pack(fill=tk.X, padx=8, pady=6)

# Кнопка загрузки изображения (показывает и заполняет табло)
load_btn = tk.Button(top_frame, text="Загрузить изображение", command=load_image)
load_btn.pack(side=tk.LEFT, padx=(0,6))

# Новая кнопка: создаём случайную линию
random_line_btn = tk.Button(top_frame, text="Случайная линия", command=create_random_line)
random_line_btn.pack(side=tk.LEFT, padx=(0,6))

# Добавлена кнопка: зигзаг по числам
zigzag_btn = tk.Button(top_frame, text="Зигзаг (по числам)", command=create_zigzag_numbers)
zigzag_btn.pack(side=tk.LEFT, padx=(0,6))

width_label = tk.Label(top_frame, text="Ширина (px):")
width_label.pack(side=tk.LEFT)
width_var = tk.StringVar()
width_entry = tk.Entry(top_frame, textvariable=width_var, width=8)
width_entry.pack(side=tk.LEFT, padx=(4,12))

duration_label = tk.Label(top_frame, text="Длительность (сек.):")
duration_label.pack(side=tk.LEFT)
duration_var = tk.StringVar(value="60")  # по умолчанию 60 секунд
duration_entry = tk.Entry(top_frame, textvariable=duration_var, width=8)
duration_entry.pack(side=tk.LEFT, padx=(4,12))

open_from_text_btn = tk.Button(top_frame, text="Открыть изображение из RGB", command=open_image_from_text)
open_from_text_btn.pack(side=tk.LEFT, padx=(0,6))

save_audio_btn = tk.Button(top_frame, text="Сохранить аудио (MP3/WAV)", command=save_audio)
save_audio_btn.pack(side=tk.LEFT, padx=(6,6))

clear_btn = tk.Button(top_frame, text="Очистить табло", command=clear_text)
clear_btn.pack(side=tk.LEFT)

text_frame = tk.Frame(root)
text_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

text_widget = tk.Text(text_frame, wrap=tk.NONE, font=("Consolas", 11))
yscroll = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
xscroll = tk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=text_widget.xview)
text_widget.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
yscroll.pack(side=tk.RIGHT, fill=tk.Y)
xscroll.pack(side=tk.BOTTOM, fill=tk.X)
text_widget.pack(fill=tk.BOTH, expand=True)

setup_clipboard_bindings(text_widget)

hint = tk.Label(root, text="Формат: по одному триплету на строку: R G B   (или R,G,B).\nЕсли вы работаете с простой таблицей чисел (каждая строка — ряд чисел), используйте кнопку 'Зигзаг (по числам)'.\nЕсли поле 'Ширина' пустое, пытаемся подобрать квадрат.\nДлительность по умолчанию — 60 секунд. Нажмите 'Сохранить аудио' для получения MP3 (или WAV — если нет pydub/ffmpeg).", anchor="w", justify=tk.LEFT, wraplength=940)
hint.pack(fill=tk.X, padx=8, pady=(0,8))

root.mainloop()

