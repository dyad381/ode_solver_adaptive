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
def dynamic(x, y):
    global step_size, coordinates
    while True:
        coordinates[0], coordinates[1] = x, y
        runge_kutta(4)
        rk4 = coordinates.copy()
        coordinates[0], coordinates[1] = x, y
        runge_kutta(5)
        rk5 = coordinates.copy()
        error = abs(rk4[1] - rk5[1])
        error = max(error, 1e-14)
        if(error > TOLERANCE):
            step_size = step_size * clamp(0.9 * (TOLERANCE/error)**(1/(p+1)), 0.1, 1.0)
            continue
        else:
            step_size = step_size * clamp(0.9 * (TOLERANCE/error)**(1/(p+1)), 0.1, 5)
            coordinates[0], coordinates[1] = x, y
            return step_size

# -------------------- BREAKER ------------------------
def function(x, y):
    return (-1+(2*x*(y**3))/(3*(x**2)*(y**2)))
    
coordinates = [1, 1]
step_size = 1
TOLERANCE = 1e-6
p = 4
bound = 30000
start = time.time()
while (coordinates[0] < bound):
    step_size = min(dynamic(coordinates[0], coordinates[1]), bound - coordinates[0])
    runge_kutta(4)
end = time.time()
print(f"Elapsed {end - start:.4f} seconds")
print(coordinates)