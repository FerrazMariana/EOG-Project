import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

FS = 500
ARQUIVO = "data\mariana\picada_dupla_1_500.csv"  # seu CSV (1 coluna)

# ---------- Limites de outlier (ajuste) ----------
LIM_INF = -0.0007
LIM_SUP = 0.0006

#LIM_INF = -0.002
#LIM_SUP = -0.001


# Filtros como na figura
LP = 15
HP_A = 0.5
ORDER_HP = 4
ORDER_LP = 4

# Plot (zoom opcional)
T0 = 12
T1 = 20



def hp_then_lp(sig, fs, hp, lp, order_hp=4, order_lp=4):
    sos_hp = butter(order_hp, hp, btype="highpass", fs=fs, output="sos")
    y = sosfiltfilt(sos_hp, sig)
    sos_lp = butter(order_lp, lp, btype="lowpass", fs=fs, output="sos")
    y = sosfiltfilt(sos_lp, y)
    return y

def remove_outliers_interp(sig, lim_inf, lim_sup):
    """
    Marca outliers (sig<lim_inf ou sig>lim_sup) como NaN e interpola linearmente.
    Mantém o mesmo tamanho do vetor, p não perdermos dados.
    """
    out = (sig < lim_inf) | (sig > lim_sup)
    sig2 = sig.copy()
    sig2[out] = np.nan

    idx = np.arange(len(sig2))
    good = np.isfinite(sig2)

    # precisa de pelo menos 2 pontos bons para interpolar
    if np.sum(good) >= 2:
        sig2[~good] = np.interp(idx[~good], idx[good], sig2[good])
    else:
        sig2 = sig.copy()  # caso extremo

    return sig2, out

# --------- Leitura ----------
x = np.genfromtxt(ARQUIVO, delimiter=",", dtype=float)
if x.ndim > 1:
    x = x[:, 0]
x = x[np.isfinite(x)]
x = x - np.mean(x)

# --------- Remove outliers por interpolação ----------
x, out_mask = remove_outliers_interp(x, LIM_INF, LIM_SUP)

print("Total de pontos em sig2:", len(x))
print("Quantidade de outliers:", out_mask.sum())

t = np.arange(len(x)) / FS
m = (t >= T0) & (t <= T1) if T1 is not None else slice(None)


# --------- Filtragens (a) e (b) ----------
y_a = hp_then_lp(x, FS, HP_A, LP, ORDER_HP, ORDER_LP)


# --------- Plot ----------
plt.figure(figsize=(12, 9))

#plt.subplot(2, 1, 1)
plt.plot(t[m], y_a[m]*1000, linewidth=2, label="0.5–15 Hz")
#plt.plot(t[m], x[m], linewidth=0.9)
plt.ylabel("Signal (mV)", fontsize=35)
plt.xlabel("Time (s)", fontsize=35)
plt.tick_params(axis='both', labelsize=35)  # aumenta os números dos eixos
plt.legend(fontsize=35)
plt.grid(True)


plt.tight_layout()
plt.show()

# --------- Salvar apenas tempo e sinal ----------
#OUT_CSV = "data/mariana/filtrado/careta.csv"

#dados = np.column_stack([t, y_a])  # tempo, sinal final (COM TODOS OS FILTROS)
#np.savetxt(OUT_CSV, dados, delimiter=",", header="t_s,sinal_filtrado", comments="")



