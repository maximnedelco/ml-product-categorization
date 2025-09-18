# Proiect de Clasificare a Produselor

Acest proiect demonstreaza un pipeline complet de machine learning pentru a clasifica titlurile de produse in categorii predefinite. Modelul este antrenat pe un set de date real si poate fi folosit pentru a prezice categoria unui produs pe baza titlului sau.

---

## Structura Proiectului

* `Notebook/Product_Category_Prediction.ipynb`: Un notebook Jupyter care contine analiza exploratorie a datelor si evaluarea detaliata a modelului. Aici vei gasi matricea de confuzie si alte metrici de performanta.
* `data/IMLP4_TASK_03-products.csv`: Setul de date utilizat pentru antrenarea modelului. Contine titluri de produse si etichetele lor de categorie.
* `src/`: Directorul care contine scripturile Python:
    * `train_model.py`: Scriptul care incarca datele, le pre-proceseaza, antreneaza modelul de clasificare si il salveaza.
    * `predict_category.py`: Un script interactiv care incarca modelul salvat si face predictii pe baza titlurilor introduse de utilizator.
* `.gitignore`: Fisierul care specifica ce fisiere trebuie ignorate de Git, cum ar fi fisierul model mare (`.pkl`).

---

## Cum sa Rulezi Proiectul

Urmeaza pasii de mai jos pentru a antrena si a rula modelul pe masina ta locala.

### Pasul 1: Instalarea Dependentelor

Asigura-te ca ai instalat bibliotecile necesare. Poti face acest lucru ruland urmatoarea comanda in terminal:

```bash
pip install pandas scikit-learn joblib
Pasul 2: Antrenarea Modelului
Ruleaza scriptul de antrenare pentru a genera fisierul model/category_model.pkl.

Bash

python src/train_model.py
Daca totul decurge corect, vei vedea mesajul: "Model trained and saved as 'model/category_model.pkl'."

Pasul 3: Utilizarea Scriptului de Predictie
Odata ce modelul este antrenat, poti folosi scriptul de predictie pentru a testa modelul.

Bash

python src/predict_category.py
Acest script te va intreba sa introduci un titlu de produs si va afisa categoria prezisa de model. Poti scrie exit pentru a opri programul.

Detalii Tehnice
Modelul: Un pipeline de machine learning care include un ColumnTransformer si un LinearSVC (Support Vector Classifier).

Pre-procesarea: Textul este vectorizat folosind TF-IDF (TfidfVectorizer), iar caracteristica title_length este scalata cu MinMaxScaler.

Evaluarea: Performanta modelului este evaluata in notebook-ul Jupyter prin crearea si analizarea unei matrici de confuzie.






The English part




Product Categorization Project
This project demonstrates a complete machine learning pipeline for classifying product titles into predefined categories. The model is trained on a real-world dataset and can be used to predict the category of a product based on its title.

Project Structure
Notebook/Product_Category_Prediction.ipynb: A Jupyter Notebook containing the exploratory data analysis and a detailed evaluation of the model's performance. You will find the confusion matrix and other key metrics here.

data/IMLP4_TASK_03-products.csv: The dataset used for training the model. It contains product titles and their corresponding category labels.

src/: The directory containing the Python scripts:

train_model.py: The script that loads the data, preprocesses it, trains the classification model, and saves it to a file.

predict_category.py: An interactive script that loads the saved model and makes predictions based on user-provided product titles.

.gitignore: A file that specifies which files should be ignored by Git, such as the large model file (.pkl).

How to Run the Project
Follow the steps below to train and run the model on your local machine.

Step 1: Install Dependencies
Make sure you have all the necessary libraries installed. You can do this by running the following command in your terminal:

Bash

pip install pandas scikit-learn joblib
Step 2: Train the Model
Run the training script to generate the model/category_model.pkl file.

Bash

python src/train_model.py
If everything runs correctly, you will see the message: "Model trained and saved as 'model/category_model.pkl'."

Step 3: Use the Prediction Script
Once the model is trained, you can use the prediction script to test it out.

Bash

python src/predict_category.py
This script will prompt you to enter a product title and will display the category predicted by the model. You can type exit to stop the program.

Technical Details
Model: A machine learning pipeline that includes a ColumnTransformer and a LinearSVC (Support Vector Classifier).

Preprocessing: The text is vectorized using TF-IDF (TfidfVectorizer), and the title_length feature is scaled using MinMaxScaler.

Evaluation: The model's performance is evaluated in the Jupyter Notebook by creating and analyzing a confusion matrix.


