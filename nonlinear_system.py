
import math
import csv
from typing import List, Tuple

# --------- Persamaan ---------
def f1(x: float, y: float) -> float:
    return x*x + x*y - 10.0

def f2(x: float, y: float) -> float:
    return y + 3.0*x*y*y - 57.0

# --------- Fungsi Iterasi (NIMx = 2 → g2A & g1B) ---------
# g1B: x_{n+1} = sqrt(10 - x_n*y_n)  (ambil akar non-negatif)
def g1B(x: float, y: float) -> float:
    val = 10.0 - x*y
    if val < 0:
        return x
    return math.sqrt(val)

# g2A (rearrangement stabil):
# y_{n+1} = 57 / (1 + 3*x_n*y_n)
def g2A(x: float, y: float) -> float:
    denom = 1.0 + 3.0*x*y
    if abs(denom) < 1e-12:
        return y
    return 57.0 / denom

# --------- Util ---------
def norm2(dx: float, dy: float) -> float:
    return math.hypot(dx, dy)

def solve_2x2(a: float, b: float, c: float, d: float,
              r1: float, r2: float) -> Tuple[float, float]:

    det = a*d - b*c
    if abs(det) < 1e-15:
        raise ZeroDivisionError("Determinant ~ 0 (matrix singular).")
    u = (r1*d - b*r2) / det
    v = (a*r2 - r1*c) / det
    return u, v

def it_jacobi(x0: float, y0: float, eps: float=1e-6, max_iter: int=500):
    logs: List[List[float]] = []
    x, y = x0, y0
    for k in range(max_iter):
        x_next = g1B(x, y)
        y_next = g2A(x, y)
        err = norm2(x_next - x, y_next - y)
        logs.append([k, x, y, x_next, y_next, err])
        x, y = x_next, y_next
        if err < eps:
            break
    return x, y, k+1, logs

def it_seidel(x0: float, y0: float, eps: float=1e-6, max_iter: int=500):
    logs: List[List[float]] = []
    x, y = x0, y0
    for k in range(max_iter):
        x_new = g1B(x, y)        
        y_new = g2A(x_new, y)     
        err = norm2(x_new - x, y_new - y)
        logs.append([k, x, y, x_new, y_new, err])
        x, y = x_new, y_new
        if err < eps:
            break
    return x, y, k+1, logs

def newton_raphson(x0: float, y0: float, eps: float=1e-6, max_iter: int=100):
    logs: List[List[float]] = []
    x, y = x0, y0
    for k in range(max_iter):
        a = 2.0*x + y          
        b = x                    
        c = 3.0*y*y              
        d = 1.0 + 6.0*x*y        
        F1, F2 = f1(x, y), f2(x, y)
        rx, ry = -F1, -F2
        dx, dy = solve_2x2(a, b, c, d, rx, ry)
        x1, y1 = x + dx, y + dy
        err = norm2(dx, dy)
        logs.append([k, x, y, x1, y1, err])
        x, y = x1, y1
        if err < eps:
            break
    return x, y, k+1, logs

def secant_method(x0: float, y0: float, eps: float=1e-6, max_iter: int=100, h: float=1e-5):
    logs: List[List[float]] = []
    x, y = x0, y0
    for k in range(max_iter):
        F1, F2 = f1(x, y), f2(x, y)
        a = (f1(x+h, y) - F1) / h   
        b = (f1(x, y+h) - F1) / h   
        c = (f2(x+h, y) - F2) / h   
        d = (f2(x, y+h) - F2) / h   
        rx, ry = -F1, -F2
        dx, dy = solve_2x2(a, b, c, d, rx, ry)
        x1, y1 = x + dx, y + dy
        err = norm2(dx, dy)
        logs.append([k, x, y, x1, y1, err])
        x, y = x1, y1
        if err < eps:
            break
    return x, y, k+1, logs

def save_csv(filename: str, rows: List[List[float]]):
    headers = ["iter", "x_n", "y_n", "x_next", "y_next", "error_norm"]
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

# --------- MAIN ---------
if __name__ == "__main__":
    x0, y0 = 1.5, 3.5
    eps = 1e-6

    xj, yj, nj, logj = it_jacobi(x0, y0, eps, max_iter=200)
    print("=== IT Jacobi (g2A & g1B) ===")
    print(f"Hasil akhir   : x = {xj:.12f}, y = {yj:.12f}")
    print(f"Jumlah iterasi: {nj}")
    print(f"Error terakhir: {logj[-1][5]:.3e}\n")

    xs, ys, ns, logs = it_seidel(x0, y0, eps, max_iter=500)
    print("=== IT Seidel (g2A & g1B) ===")
    print(f"Hasil akhir   : x = {xs:.12f}, y = {ys:.12f}")
    print(f"Jumlah iterasi: {ns}")
    print(f"Error terakhir: {logs[-1][5]:.3e}\n")

    xn, yn, nn, logn = newton_raphson(x0, y0, eps)
    print("=== Newton–Raphson ===")
    print(f"Hasil akhir   : x = {xn:.12f}, y = {yn:.12f}")
    print(f"Jumlah iterasi: {nn}")
    print(f"Error terakhir: {logn[-1][5]:.3e}\n")

    xq, yq, nq, logq = secant_method(x0, y0, eps)
    print("=== Secant (Jacobian numerik) ===")
    print(f"Hasil akhir   : x = {xq:.12f}, y = {yq:.12f}")
    print(f"Jumlah iterasi: {nq}")
    print(f"Error terakhir: {logq[-1][5]:.3e}\n")

    SAVE_CSV = False #Ubah ke True jika ingin save file csv
    if SAVE_CSV:
        save_csv("IT_Jacobi_g2A_g1B.csv", logj)
        save_csv("IT_Seidel_g2A_g1B.csv", logs)
        save_csv("Newton_Raphson.csv", logn)
        save_csv("Secant.csv", logq)
        print("Log iterasi disimpan sebagai CSV di folder script.")
