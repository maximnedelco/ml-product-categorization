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

    
    user_input = pd.DataFrame([{
        "product title": product_title, 
        "title_length": title_length    
    }])

    # Efectueaza predictia
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}\n"+"-"*40)