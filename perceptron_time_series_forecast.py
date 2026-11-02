import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================================================================
# НАСТРОЙКИ - МОЖНО МЕНЯТЬ
# =============================================================================
EXCEL_NAME   = "TATN.xlsx"   # файл с котировками (ищем в папке скрипта и в текущей)
COLUMN_NAME  = None          # столбец с данными (None = первый числовой)

N_IN    = 10    # количество входных точек
N_OUT   = 5     # количество выходных точек
N_CROSS = 3     # количество перекрёстных точек (8–10 входят и во вход, и в выход)

DELTA   = 0.05  # нормировка ряда в [DELTA, 1-DELTA] для сигмоиды
LR      = 0.01  # шаг градиентного спуска λ
EPOCHS  = 1000  # количество эпох обучения

SPLIT_PERCENT = 80       # процент окон, используемых для обучения на части выборки (0–100)
FUTURE_STEPS  = 100       # сколько точек экстраполировать в будущее
FIG_SIZE      = (12, 10)  # размер окна с графиками (ширина, высота)
VERBOSE       = True      # выводить прогресс по эпохам при обучении


# =============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# =============================================================================

def find_excel(name: str) -> str:
    """Ищем файл TATN.xlsx / TATN.XLSX в папке скрипта и в текущей."""
    base, ext = os.path.splitext(name)
    for folder in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        for variant in (name, base + ".XLSX", base + ".xlsx"):
            path = os.path.join(folder, variant)
            if os.path.isfile(path):
                return path
    return name


def load_series(path: str, column: str | None = None) -> np.ndarray:
    """
    Загружаем одномерный ряд котировок из Excel-файла.
    Если столбец не указан - берём первый числовой.
    """
    df = pd.read_excel(path)
    if column is not None:
        series = df[column]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("В файле нет числовых столбцов.")
        series = df[numeric_cols[0]]
    return series.dropna().to_numpy(dtype=float)


# =============================================================================
# 2. НОРМИРОВКА
# Нормируем ряд в [delta, 1-delta], чтобы сигмоида работала в зоне
# с адекватной скоростью изменения (вдали от насыщения у 0 и 1).
# =============================================================================

def normalize(data: np.ndarray, delta: float = 0.05
              ) -> tuple[np.ndarray, float, float]:
    """
    Нормировка ряда в [delta, 1 - delta].
    Формула: z = delta + (x - x_min) / (x_max - x_min) * (1 - 2*delta)
    Возвращает: нормированный массив, x_min, x_max (нужны для денормировки).
    """
    x_min = data.min()
    x_max = data.max()
    z = (data - x_min) / (x_max - x_min)           # приводим к [0, 1]
    z = delta + z * (1.0 - 2.0 * delta)             # сжимаем к [delta, 1-delta]
    return z, x_min, x_max


def denormalize(z: np.ndarray, x_min: float, x_max: float,
                delta: float = 0.05) -> np.ndarray:
    """
    Обратная нормировка: из [delta, 1-delta] -> исходная шкала.
    Формула: x = (z - delta) / (1 - 2*delta) * (x_max - x_min) + x_min
    """
    return (z - delta) / (1.0 - 2.0 * delta) * (x_max - x_min) + x_min


# =============================================================================
# 3. ПОСТРОЕНИЕ ДАТАСЕТА
# Скользящее окно: на вход n_in точек, на выход n_out точек,
# из них n_cross перекрёстных (одновременно во входе и в выходе),
# остальные - будущие. Окно сдвигается на 1 точку вправо.
#
# Пример (1-индексация, n_in=10, n_out=5, n_cross=3):
#   вход  -> точки 1-10
#   выход -> точки 8-13  (8,9,10 - перекрёстные; 11,12,13 - будущие)
# =============================================================================

def build_dataset(series: np.ndarray,
                  n_in: int = 10,
                  n_out: int = 5,
                  n_cross: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Строим датасет методом скользящего окна.

    Для окна, начинающегося в позиции k (0-индексация):
      - входной вектор x⃗:  series[k : k + n_in]
      - целевой вектор t⃗:  series[k + n_in - n_cross : k + n_in - n_cross + n_out]

    Пример при n_in=10, n_out=5, n_cross=3:
      вход  → точки k+1 … k+10
      выход → точки k+8 … k+12  (k+8,k+9,k+10 — перекрёстные)
    """
    total = n_in + n_out - n_cross  # ширина одного окна = 12
    X_list, T_list = [], []
    k = 0
    while k + total <= len(series):
        x_vec = series[k : k + n_in]
        t_start = k + n_in - n_cross
        t_vec = series[t_start : t_start + n_out]
        X_list.append(x_vec)
        T_list.append(t_vec)
        k += 1  # сдвигаем окно на 1 точку
    # X: (n_samples, n_in),  T: (n_samples, n_out)
    return np.stack(X_list), np.stack(T_list)


def reconstructed_series(series_len: int, X_out: np.ndarray,
                         n_in: int, n_out: int, n_cross: int) -> np.ndarray:
    """
    Восстанавливаем ряд по выходам перцептрона: каждый крестик в своём подлинном
    значении. Для каждого индекса ряда усредняем все предсказания из окон,
    в которых этот индекс входит в выход.
    X_out: (n_samples, n_out) - выходы модели по всем окнам.
    Возвращает массив длины series_len (NaN там, где нет предсказаний).
    """
    rec = np.zeros(series_len, dtype=float)
    cnt = np.zeros(series_len)
    n_samples = X_out.shape[0]
    for d in range(n_samples):
        t_start = d + n_in - n_cross
        for j in range(n_out):
            idx = t_start + j
            if idx < series_len:
                rec[idx] += X_out[d, j]
                cnt[idx] += 1
    for i in range(series_len):
        if cnt[i] > 0:
            rec[i] /= cnt[i]
        else:
            rec[i] = np.nan
    return rec


# =============================================================================
# 4. ФУНКЦИЯ АКТИВАЦИИ - СИГМОИДА
# σ(S) = 1 / (1 + e^(-S))
# Производная: σ'(S) = σ(S) * (1 - σ(S)) = x * (1 - x)
# =============================================================================

def sigmoid(S: np.ndarray) -> np.ndarray:
    """σ(S) = 1 / (1 + e^(-S))"""
    return 1.0 / (1.0 + np.exp(-S))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    """
    Производная сигмоиды через её выход x = σ(S):
    [∂f/∂S] = x · (1 - x)
    """
    return x * (1.0 - x)


# =============================================================================
# 5. ПЕРЦЕПТРОН
# Однослойный перцептрон (входной слой -> выходной слой) без скрытых слоёв.
# Матрица весов W имеет размерность (n_in+1) x n_out, где +1 - нейрон смещения.
#
# Прямой проход:
#   S = X_tilde @ W,   X_out = σ(S)
#
# Целевая функция:
#   ε_d = Σ_j (x_j^d - t_j^d)^2
#   E   = Σ_d ε_d
#
# Градиент (правило цепочки):
#   ∂E/∂W = X_tilde^T @ Δ,   Δ = (X_out - T) * σ'(X_out)
#
# Шаг градиентного спуска:
#   W_new = W_old - λ * ∂E/∂W
# =============================================================================

class Perceptron:
    """
    Однослойный перцептрон (два слоя нейронов: входной и выходной).
    Нейрон смещения добавляется к входному слою.
    """

    def __init__(self, n_in: int, n_out: int, lr: float = 0.01) -> None:
        self.n_in  = n_in
        self.n_out = n_out
        self.lr    = lr

        # Инициализируем W произвольными значениями (из [-0.5, 0.5]).
        # Размерность: (n_in + 1) × n_out.
        # Строка 0 - веса нейрона смещения.
        # Строки 1..n_in - веса входных нейронов.
        rng = np.random.default_rng(seed=42)
        self.W = rng.uniform(-0.5, 0.5, size=(n_in + 1, n_out))

    # ------------------------------------------------------------------
    # Нейрон смещения: добавляем столбец единиц к матрице входов.
    # ------------------------------------------------------------------
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_samples, n_in) -> X_tilde: (n_samples, n_in + 1)
        Первый столбец - единицы (нейрон смещения).
        """
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    # ------------------------------------------------------------------
    # Прямой проход
    # S = X̃ · W      (формула конспекта: S_j̄ = Σᵢ xᵢ·Wᵢⱼ̄ + 1·W₁ᵢⱼ̄)
    # X_out = σ(S)   (формула конспекта: xⱼ = fʲ(Sⱼ))
    # ------------------------------------------------------------------
    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Возвращает (X_out, X_tilde):
          X_out   - выходные значения нейронов, (n_samples, n_out)
          X_tilde - расширенный вход с bias,    (n_samples, n_in + 1)
        X_tilde возвращаем, чтобы не пересчитывать его в train().
        """
        X_tilde = self._add_bias(X)         # X̃: добавляем нейрон смещения
        S       = X_tilde @ self.W          # S = X̃ · W,  (n_samples, n_out)
        X_out   = sigmoid(S)                # xⱼ = σ(Sⱼ), (n_samples, n_out)
        return X_out, X_tilde

    # ------------------------------------------------------------------
    # Целевая функция: E = Σ_d ε_d,  ε_d = Σ_j (x_j^d - t_j^d)^2
    # ------------------------------------------------------------------
    def loss(self, X_out: np.ndarray, T: np.ndarray) -> float:
        """E = Σd Σⱼ (xⱼᵈ - tⱼᵈ)²"""
        return float(np.sum((X_out - T) ** 2))

    # ------------------------------------------------------------------
    # Шаг обучения (батч-градиентный спуск):
    #   ∂E/∂W = X_tilde^T @ Δ
    #   W_new = W_old - λ * ∂E/∂W
    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, T: np.ndarray,
              epochs: int = 1000, verbose: bool = True) -> list[float]:
        """
        Градиентный спуск по лекции: для каждого элемента датасета — прямой прогон,
        затем суммируем ∂εd/∂W по всем d и делаем один шаг W -= λ·∂E/∂W.
        Здесь всё датасет обрабатывается векторно за один раз (без мини-батчей).
        Возвращает историю значений E по эпохам.
        """
        history = []

        for epoch in range(epochs):

            # --- прямой прогон по всему датасету (x⃗ᵢ, S⃗ⱼ, x⃗ⱼ) ---
            X_out, X_tilde = self.forward(X)  # X_out = xⱼ, (n_samples, n_out)

            # --- целевая функция E = Σd εd ---
            E = self.loss(X_out, T)
            history.append(E)

            # --- матрица Δ: для каждого d и j̄ — (xⱼ̄ᵈ - tⱼ̄ᵈ) · [∂f/∂S] ---
            Delta = (X_out - T) * sigmoid_grad(X_out)  # (n_samples, n_out)

            # --- ∂E/∂W = Σd ∂εd/∂W в матричном виде: X̃ᵀ · Δ ---
            grad_W = X_tilde.T @ Delta  # (n_in+1, n_out)

            # --- шаг градиентного спуска: W_new = W_old - λ · ∂E/∂W ---
            self.W -= self.lr * grad_W

            if verbose and (epoch % max(1, epochs // 10) == 0
                            or epoch == epochs - 1):
                print(f"Epoch {epoch + 1:4d}/{epochs},  E = {E:.6f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает выходные значения сети для входной матрицы X."""
        X_out, _ = self.forward(X)
        return X_out


# =============================================================================
# 6. ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def main() -> None:
    # -- путь к файлу TATN.xlsx / TATN.XLSX --
    excel_path = find_excel(EXCEL_NAME)
    if not os.path.isfile(excel_path):
        print(f"Файл {excel_path} не найден.")
        return
    print(f"Файл данных: {excel_path}")
    series = load_series(excel_path, column=COLUMN_NAME)
    print(f"Загружено точек: {len(series)}")

    # -- нормировка в [DELTA, 1-DELTA] --
    series_norm, x_min, x_max = normalize(series, delta=DELTA)

    # -- построение датасета --
    # X: (n_samples, N_IN),  T: (n_samples, N_OUT)
    X, T = build_dataset(series_norm, n_in=N_IN, n_out=N_OUT, n_cross=N_CROSS)
    print(f"Датасет: {X.shape[0]} окон  |  X: {X.shape}  |  T: {T.shape}")

    # -- создание и обучение перцептрона (модель на всём датасете) --
    model = Perceptron(n_in=N_IN, n_out=N_OUT, lr=LR)

    # матрица весов ДО обучения
    print("\n=== МАТРИЦА ВЕСОВ model ДО ОБУЧЕНИЯ ===")
    print(model.W)

    history = model.train(X, T, epochs=EPOCHS, verbose=VERBOSE)
    print(f"\nФинальная ошибка E = {history[-1]:.6f}")

    # матрица весов ПОСЛЕ обучения
    print("\n=== МАТРИЦА ВЕСОВ model ПОСЛЕ ОБУЧЕНИЯ ===")
    print(model.W)

    # -- восстановление ряда по первой модели --
    X_out_all = model.predict(X)
    rec_norm_full = reconstructed_series(len(series_norm), X_out_all, N_IN, N_OUT, N_CROSS)
    rec_full = denormalize(rec_norm_full, x_min, x_max, delta=DELTA)

    # -- обучение на части выборки (вторая модель) --
    split = int(X.shape[0] * SPLIT_PERCENT / 100)
    model_part = Perceptron(n_in=N_IN, n_out=N_OUT, lr=LR)

    # матрица весов второй модели ДО обучения
    print("\n=== МАТРИЦА ВЕСОВ model_part ДО ОБУЧЕНИЯ ===")
    print(model_part.W)

    model_part.train(X[:split], T[:split], epochs=EPOCHS, verbose=VERBOSE)
    X_out_part = model_part.predict(X)
    rec_norm_part = reconstructed_series(len(series_norm), X_out_part, N_IN, N_OUT, N_CROSS)
    rec_part = denormalize(rec_norm_part, x_min, x_max, delta=DELTA)

    # матрица весов второй модели ПОСЛЕ обучения
    print("\n=== МАТРИЦА ВЕСОВ model_part ПОСЛЕ ОБУЧЕНИЯ ===")
    print(model_part.W)

    # -- экстраполяция в будущее на FUTURE_STEPS точек --
    series_ext = series_norm.tolist()
    future_norm_list: list[float] = []
    for _ in range(FUTURE_STEPS):
        last_input = np.array(series_ext[-N_IN:]).reshape(1, -1)
        pred_segment = model.predict(last_input)[0]
        next_point = float(pred_segment[-1])
        future_norm_list.append(next_point)
        series_ext.append(next_point)

    future_norm = np.array(future_norm_list)
    future_real = denormalize(future_norm, x_min, x_max, delta=DELTA)
    future_idx = np.arange(len(series), len(series) + FUTURE_STEPS)

    # -- визуализация: 3 графика --
    fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE, sharex=False)
    t_idx = np.arange(len(series))

    # 1. Исходные данные - синие крестики; модель - красная линия
    axes[0].plot(t_idx, series, "bx", markersize=5, label="исходные данные", alpha=0.9)
    valid = ~np.isnan(rec_full)
    axes[0].plot(t_idx[valid], rec_full[valid], "r-", label="восстановление (вся выборка)", alpha=0.9)
    axes[0].set_title("1. Исходные данные и восстановление ряда (обучение на всём датасете)")
    axes[0].set_xlabel("Время")
    axes[0].legend()
    axes[0].grid(True)

    # 2. Обучение на части выборки
    axes[1].plot(t_idx, series, "bx", markersize=5, label="исходные данные", alpha=0.9)
    valid1 = ~np.isnan(rec_part)
    axes[1].plot(t_idx[valid1], rec_part[valid1], "r-", label="восстановление (обучение на части)", alpha=0.9)
    boundary_idx = split + N_IN - N_CROSS - 1
    axes[1].axvline(x=boundary_idx, color="gray", linestyle="--",
                    label="граница обучающей выборки")
    axes[1].set_title("2. Исходные данные и восстановление (обучение на части выборки)")
    axes[1].set_xlabel("Время")
    axes[1].legend()
    axes[1].grid(True)

    # 3. Модель на всём датасете + экстраполяция
    axes[2].plot(t_idx, series, "bx", markersize=5, label="исходные данные", alpha=0.9)
    valid2 = ~np.isnan(rec_full)
    axes[2].plot(t_idx[valid2], rec_full[valid2], "r-", label="модель на всём датасете", alpha=0.9)
    axes[2].plot(future_idx, future_real, "r--", label="экстраполяция (прогноз)", alpha=0.9)
    axes[2].set_title("3. Исходные данные и экстраполяция в будущее")
    axes[2].set_xlabel("Время")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    os.makedirs(images_dir, exist_ok=True)
    fig_path = os.path.join(images_dir, "results_overview.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"График сохранён: {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()



