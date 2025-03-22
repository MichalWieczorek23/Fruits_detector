# Fruit Detector 🍏🍊🍓

## Opis projektu

**Fruit Detector** to projekt oparty na podejściu **RCNN**, pozwalający na wykrywanie i klasyfikację owoców. Implementacja umożliwia użycie różnych modeli sieci konwolucyjnych oraz klasyfikatorów.

Obecnie projekt wykorzystuje **EfficientNetB0** do ekstrakcji cech, choć planowane są eksperymenty z innymi modelami. Warstwa klasyfikatora może bazować na różnych podejściach, takich jak **SVM** oraz **MLP**, z czego **SVC** zostało już zaimplementowane.

### Potencjalne zastosowania

Fruit Detector może znaleźć zastosowanie w **inteligentnym rolnictwie**, jako część większego systemu.

---

## Technologie

Projekt wykorzystuje następujące technologie i biblioteki:

- **Keras** – budowa i trenowanie modeli sieci neuronowych
- **PIL (Pillow)** – obsługa i przetwarzanie obrazów
- **OpenCV (cv2)** – operacje na obrazach
- **Scikit-learn (sklearn)** – SVM, GridSearch
- **NumPy** – operacje na macierzach i przetwarzanie danych

---


## Planowane ulepszenia

- Dodanie klasyfikatora bazującego na MLP
- Możliwość wyboru modelu ekstrakcji cech (np. ResNet, MobileNet)
- Optymalizacja wydajności i szybkości predykcji
- Rozwój podejścia RCNN do bardziej zaawansowanych metod takich jak Fast-RCNN, Faster-RCNN, Mask-RCNN

---

