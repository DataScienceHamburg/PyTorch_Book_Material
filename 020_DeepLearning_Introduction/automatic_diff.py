# %% Pakete importieren
import torch
import matplotlib.pyplot as plt

# %% 1. Daten und Modell initialisieren
# Wir wollen die lineare Funktion y = 3x + 2 annähern.
# X_true ist unser Eingabe-Tensor (x).
X_true = torch.arange(1, 10, dtype=torch.float32).unsqueeze(1)
# y_true ist unser Ziel-Tensor (y).
y_true = 3 * X_true + 2

#%% visualize data
plt.scatter(X_true, y_true)
plt.plot(X_true, 3 * X_true + 2, 'r-', label='Wahre Regressionsgerade')
plt.xlabel('Unabhängige Variable (X_true)')
plt.ylabel('Abhängige Variable (y_true)')
plt.legend()
plt.show()
#%%
# Unsere lernbaren Parameter, die wir optimieren wollen.
# requires_grad=True ist hier der Schlüssel!
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f"Gewichte 'w' (zufällig initialisiert): {w.item():.4f}")
print(f"Bias 'b' (zufällig initialisiert): {b.item():.4f}")

# Definiere eine Lernrate, die bestimmt, wie groß unsere Schritte sind.
LEARNING_RATE = 0.01
EPOCHS = 1000

# %% 2. Der Trainingsprozess in einer Schleife
# Wir trainieren das Modell über 100 Epochen (Durchläufe).
w_list = []
b_list = []
loss_list = []
for epoch in range(EPOCHS):
    
    # 3. Forward Pass: Die Vorhersage berechnen
    y_pred = w * X_true + b
    
    # 4. Verlust berechnen
    # Wir verwenden den Mean Squared Error (MSE), der den quadratischen
    # Abstand zwischen Vorhersage und Zielwert misst.
    loss = torch.mean((y_pred - y_true)**2)
    
    # 5. Backward Pass: Gradienten berechnen
    # Dies ist der entscheidende Schritt, der die Gradienten für w und b ermittelt.
    loss.backward()
    
    # 6. Parameter-Update
    # Wir aktualisieren die Parameter in die entgegengesetzte Richtung des Gradienten,
    # um den Verlust zu minimieren. Hier verwenden wir den Kontext 'with torch.no_grad()',
    # um sicherzustellen, dass diese Aktualisierungen nicht Teil des Rechengraphen werden.
    with torch.no_grad():
        w -= LEARNING_RATE * w.grad
        b -= LEARNING_RATE * b.grad
        w_list.append(w.item())
        b_list.append(b.item())
        loss_list.append(loss.item())
    # 7. Gradienten zurücksetzen
    # Nach dem Update müssen wir die Gradienten manuell auf 0 setzen.
    # Sonst würden sie sich in der nächsten Iteration aufsummieren.
    w.grad.zero_()
    b.grad.zero_()
    
    # f) Fortschritt ausgeben (optional)
    if (epoch + 1) % 10 == 0:
        print(f"Epoche [{epoch+1}/{EPOCHS}], Verlust: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")

#%% plot w, b, and loss
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

ax1.plot(w_list, label='w')
ax1.set_title('Gewicht')
ax1.set_xlabel('Epoche [-]')
ax1.set_ylabel('Gewicht w [-]')
ax1.legend()

ax2.plot(b_list, label='b')
ax2.set_title('Bias')
ax2.set_xlabel('Epoche [-]')
ax2.set_ylabel('Bias b [-]')
ax2.legend()

ax3.plot(loss_list, label='loss')
ax3.set_title('Verlust')
ax3.set_xlabel('Epoche [-]')
ax3.set_ylabel('Verlust [-]')
ax3.set_ylim(0, 1)  # Set vertical range from 0 to 10
ax3.legend()

plt.tight_layout()
plt.show()

# %% 3. Endgültiges Ergebnis ausgeben
print("\n--- 3. Endergebnis nach dem Training ---")
print(f"Endgültige Gewichte 'w': {w.item():.4f}")
print(f"Endgültiger Bias 'b': {b.item():.4f}")
print(f"Erwartete Werte: w=3, b=2")

