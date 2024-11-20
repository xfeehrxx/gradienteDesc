def grad_18(input, output, w1, w2, b):
    dL_dw1 = 0
    dL_dw2 = 0
    dL_db = 0
    for i, out in zip(input, output):
        pred = w1 * i**18 + w2 * i + b
        dL_dw1 += -2 * (out - pred) * i**18
        dL_dw2 += -2 * (out - pred) * i
        dL_db += -2 * (out - pred)
    return dL_dw1, dL_dw2, dL_db

def descent(init, lr, epochs, input, output, gradient, tol=10**-6):
    count = 0
    for _ in range(epochs):
        count += 1
        grad = gradient(input, output, *init)
        grad = [init[i] - lr * grad[i] for i in range(len(grad))]
        err = sum((output[i] - (init[0] * input[i]**18 + init[1] * input[i] + init[2]))**2 for i in range(len(input)))
        if err < tol:
            break
        init = grad
    return grad, err, count

X = [i / 10 for i in range(1, 5)] 
y = [i**18 + i + 1 for i in X]

result = descent((2, 2, 2), 0.2, 10000, X, y, grad_18)

print("Parâmetros finais:", result[0])
print("Erro final:", result[1])
print("Número de iterações:", result[2])
