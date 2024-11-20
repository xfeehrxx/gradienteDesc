import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def grad_logistic(input, output, w1, b):
    dL_dw1 = 0
    dL_db = 0
    for i, out in zip(input, output):
        pred = sigmoid(w1 * i + b)
        dL_dw1 += - (out - pred) * i
        dL_db += - (out - pred)
    return dL_dw1, dL_db

def descent(init, lr, epochs, input, output, gradient, tol=10**-6):
    count = 0
    for _ in range(epochs):
        count += 1
        grads = gradient(input, output, *init)
        init = [init[i] - lr * grads[i] for i in range(len(grads))]
        err = -sum(out * math.log(sigmoid(init[0] * i + init[1])) + (1 - out) * math.log(1 - sigmoid(init[0] * i + init[1])) 
                   for i, out in zip(input, output))
        if err < tol:
            break
    return init, err, count

def evaluate(y_true, y_pred):
    y_pred_classes = [1 if pred >= 0.5 else 0 for pred in y_pred]

    correct = sum(1 for true, pred in zip(y_true, y_pred_classes) if true == pred)
    accuracy = correct / len(y_true)
    
    tp = sum(1 for true, pred in zip(y_true, y_pred_classes) if true == pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred_classes) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred_classes) if true == 1 and pred == 0)
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return accuracy, f1

X = [i / 10 for i in range(1, 11)]
y = [1 if i >= 0.5 else 0 for i in X] 

init_params = [0.5, 0.5]
result = descent(init_params, 0.6, 10000, X, y, grad_logistic)

w1, b = result[0]
predictions = [sigmoid(w1 * i + b) for i in X]

accuracy, f1 = evaluate(y, predictions)

print("Parâmetros finais:", result[0])
print("Erro final:", result[1])
print("Número de iterações:", result[2])
print("Acurácia:", accuracy)
print("F1-Score:", f1)
