print('Question 1: Euler Method')

import numpy as np

def derivative(t, y):
    return t - y**2

t0 = 0
y0 = 1

t_end = 2
h = 0.2 

t = np.arange(t0, t_end + h, h)
y = np.zeros_like(t)
y[0] = y0

for i in range(1, len(t)):
    y[i] = y[i-1] + h * derivative(t[i-1], y[i-1])

print("t\ty")
for i in range(len(t)):
    print(f"{t[i]:.2f}\t{y[i]:.4f}")





print('Question 2: Runge-Kutta Method')

def runge_kutta_4th_order(f, y0, t0, h, t_end):

    t_values = [t0]
    y_values = [y0]
    t = t0

    while t < t_end:
        k1 = h * f(t, y_values[-1])
        k2 = h * f(t + h/2, y_values[-1] + k1/2)
        k3 = h * f(t + h/2, y_values[-1] + k2/2)
        k4 = h * f(t + h, y_values[-1] + k3)

        y_next = y_values[-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        t_next = t + h

        t_values.append(t_next)
        y_values.append(y_next)
        t = t_next

    return list(zip(t_values, y_values))

if __name__ == '__main__':
    
    f = lambda t, y: t - y**2

    y0 = 1.0  
    t0 = 0  
    h = 0.2   
    t_end = 1.8 

    solution = runge_kutta_4th_order(f, y0, t0, h, t_end)

    print("t\t\ty")
    for t, y in solution:
        print(f"{t:.2f}\t\t{y:.6f}")