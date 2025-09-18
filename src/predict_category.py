# Incarcarea librariilor
import joblib
import pandas as pd

# Incarca modelul
model = joblib.load("model/category_model.pkl")
print("Model loaded succesfully!")
print("Type 'exit' at any point to stop.\n")

# Bucla infinita pentru a solicita continuu titluri de produse de la utilizator
while True:
    product_title = input("Enter title:")
    if product_title.lower() == "exit":
        print("Exiting...")
        break

# Calculam lungimea titlului introdus de utilizator
    title_length = len(product_title)

# Cream un DataFrame pandas din datele de intrare ale utilizatorului
# Asiguram ca numele coloanelor corespund cu cele folosite in timpul antrenarii
    user_input = pd.DataFrame([{
        "product title": product_title, 
        "title_length": title_length    
    }])

# Efectuam predictia folosind modelul incarcat si datele de intrare
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}\n"+"-"*40)