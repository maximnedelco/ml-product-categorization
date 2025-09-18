import joblib
import pandas as pd

# Incarca modelul
model = joblib.load("model/category_model.pkl")
print("Model loaded succesfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    product_title = input("Enter title:")
    if product_title.lower() == "exit":
        print("Exiting...")
        break

    title_length = len(product_title)

    # Creeaza DataFrame-ul de intrare pentru predictie
    # Asigura-te ca numele coloanelor sunt exact la fel ca cele folosite la antrenare
    user_input = pd.DataFrame([{
        "product title": product_title,  # Corectat numele coloanei
        "title_length": title_length    # Corectat numele coloanei
    }])

    # Efectueaza predictia
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}\n"+"-"*40)