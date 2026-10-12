#imports
import time
import statistics

#helper functions
def clamp(val, lo, hi):
    return max(lo, min(val, hi))

#euler: yn = yn-1 + h*y'(xn-1,yn-1)
def euler():
    x, y = coordinates[0], coordinates[1]
    coordinates[1] = y + step_size*function(x, y)
    coordinates[0] += step_size
    
#improved euler: yn = yn-1 + (h/2)(y'(xn,euler(yn-1)) + y'(xn-1,yn-1))
def improved_euler():
    x, y = coordinates[0], coordinates[1]
    coordinates[1] = y + step_size*function(x, y)
    coordinates[0] += step_size
    coordinates[1] = y + (step_size*0.5)*(function(x, y) + function(coordinates[0], coordinates[1]))
def runge_kutta(steps):
    if(steps < 2):
        return None
    x, y = coordinates[0], coordinates[1]
    klist = []
    klist.append(step_size*function(x, y))
    for i in range(steps-2):
        klist.append(step_size*function(x + (step_size/2), y + klist[i]/2))
    klist.append(step_size*function(x + step_size, y + klist[-1]))
    coordinates[0] += step_size
    coordinates[1] += (1/(2*(len(klist)-1)))*(2*sum(klist) - klist[0] - klist[-1])
#two rk4 half-steps: the comparison for adaptive rk4
def rk4_halves():
    global step_size
    h = step_size
    step_size = h/2
    runge_kutta(4)
    runge_kutta(4)
    step_size = h
def dynamic(x, y, method, compare, order):
    global step_size, coordinates
    while True:
        coordinates[0], coordinates[1] = x, y
        method()
        result = coordinates.copy()
        coordinates[0], coordinates[1] = x, y
        compare()
        check = coordinates.copy()
        error = abs(result[1] - check[1]) / (1 + abs(result[1]))
        error = max(error, 1e-14)
        if(error > TOLERANCE):
            step_size = step_size * clamp(0.9 * (TOLERANCE/error)**(1/(order+1)), 0.1, 1.0)
            continue
        else:
            coordinates[0], coordinates[1] = result[0], result[1]
            step_size = step_size * clamp(0.9 * (TOLERANCE/error)**(1/(order+1)), 0.1, 5)
            return step_size

# -------------------- BREAKER ------------------------
def function(x, y):
    return 1 + y**2
    
coordinates = [1, 1]
step_size = 1
TOLERANCE = 1e-6
bound = 1.7
start = time.time()
while (coordinates[0] < bound):
    step_size = min(step_size, bound - coordinates[0])
    dynamic(coordinates[0], coordinates[1], lambda: runge_kutta(4), rk4_halves, 4)
end = time.time()
print(f"Elapsed {end - start:.4f} seconds")
print(coordinates)

# -------------------- BENCHMARK: time vs error ------------------------
import math
import matplotlib.pyplot as plt

ORIGINS = [(0, 0), (0, 1), (-1, -2), (1, 0.5)]   # starting points (x0, y0)
END_FRACTION = 0.99                               # solve 99% of the way to the blow-up point
END = None                                        # set per starting point below
STEP_SIZES = [10**(-k/4) for k in range(4, 21)]   # fixed-step runs: h = 0.1 ... 0.00001
TOLERANCES = [10**(-k/2) for k in range(4, 17)]   # adaptive runs: tol = 1e-2 ... 1e-8
REPEATS = 3                                       # time each run 3 times, keep the fastest

METHODS = [
    # name,            method,                    compare (for adaptive),    order
    ("Euler",          euler,                     improved_euler,            1),
    ("Improved Euler", improved_euler,            lambda: runge_kutta(4),    2),
    ("RK4",            lambda: runge_kutta(4),    rk4_halves,                4),
]
COLORS = {"Euler": "#2a78d6", "Improved Euler": "#eb6834", "RK4": "#1baf7a"}

# exact solution of y' = 1 + y^2 through (x0, y0): y = tan(x + C)
def exact(x, x0, y0):
    return math.tan(x + math.atan(y0) - x0)

# the solution blows up where x + C = pi/2
def blowup(x0, y0):
    return x0 + math.pi/2 - math.atan(y0)

def run_fixed(method, x0, y0, h):
    global coordinates, step_size
    coordinates = [x0, y0]
    while coordinates[0] < END:
        step_size = min(h, END - coordinates[0])
        method()
    return coordinates[1]

def run_adaptive(method, compare, order, x0, y0, tol):
    global coordinates, step_size, TOLERANCE
    TOLERANCE = tol
    coordinates = [x0, y0]
    step_size = 1
    while coordinates[0] < END:
        step_size = min(step_size, END - coordinates[0])
        dynamic(coordinates[0], coordinates[1], method, compare, order)
    return coordinates[1]

# runs one solve, returns (fastest time in seconds, relative error at END)
def measure(solve, x0, y0):
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        try:
            y_end = solve()
        except (OverflowError, ZeroDivisionError):
            return None
        best = min(best, time.perf_counter() - t0)
    true = exact(END, x0, y0)
    err = abs(y_end - true) / abs(true)
    if not math.isfinite(err) or err == 0:
        return None
    return best, err

saved_tol = TOLERANCE
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, (x0, y0) in zip(axes.flat, ORIGINS):
    END = x0 + END_FRACTION * (blowup(x0, y0) - x0)
    for name, method, compare, order in METHODS:
        for adaptive in (False, True):
            points = []
            if adaptive:
                for tol in TOLERANCES:
                    r = measure(lambda: run_adaptive(method, compare, order, x0, y0, tol), x0, y0)
                    if r: points.append(r)
            else:
                for h in STEP_SIZES:
                    r = measure(lambda: run_fixed(method, x0, y0, h), x0, y0)
                    if r: points.append(r)
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    color=COLORS[name], linewidth=2,
                    linestyle="-" if adaptive else "--",
                    marker="o" if adaptive else "s", markersize=5,
                    label=f"{name} ({'adaptive' if adaptive else 'fixed step'})")
            print(f"({x0}, {y0}) {name:15s} {'adaptive' if adaptive else 'fixed':8s} done")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Start (x0, y0) = ({x0}, {y0}), solved to x = {END:.3f}")
    ax.set_xlabel("Computation time (s)")
    ax.set_ylabel("Relative error at end point")
    ax.grid(True, which="major", color="#dddddd", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
fig.suptitle("Computation time vs error: y' = 1 + y^2  (solved to 99% of the blow-up)")
fig.tight_layout(rect=(0, 0.07, 1, 0.97))
fig.savefig("time_vs_error.png", dpi=150)
TOLERANCE = saved_tol
plt.show()
