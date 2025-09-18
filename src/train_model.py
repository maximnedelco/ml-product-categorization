# Importam librariile necesare
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
import joblib

#Incracam datele din fisierul CSV in DataFrame
df = pd.read_csv("data/IMLP4_TASK_03-products.csv")
print(df.head())
print(df.info())

# Curatam si simplificam numele coloanelor, eliminand spatiile
df.columns = df.columns.str.lower().str.strip()


print(df.info())
df.isna().sum()
df.dropna(subset= ["product title","category label"],inplace=True)


# Afisam distributia initiala a categoriilor
print(df["category label"].value_counts())

# Cream un dictionar pentru a unifica categoriile
category_map = {
    'Fridges': 'Fridge Freezers',
    'fridge': 'Fridge Freezers',
    'Fridge Freezers': 'Fridge Freezers',
    'Mobile Phone': 'Mobile Phones',
    'Mobile Phones': 'Mobile Phones',
    'CPU': 'CPUs',
    'CPUs': 'CPUs',
    'TVs': 'TVs',
    'Freezers': 'Freezers',
    'Washing Machines': 'Washing Machines',
    'Dishwashers': 'Dishwashers',
    'Digital Cameras': 'Digital Cameras',
    'Microwaves': 'Microwaves'
}


# Aplicam dicitionarul asupra categoriei
df["category"] = df["category label"].map(category_map).fillna(df['category label'])


print(df["category"].value_counts())

# Convertim coloanele la tipuri de date mai eficiente pentru memorie
print(df.dtypes)
df["category"] = df["category"].astype("category")
df["product title"] = df["product title"].astype("string")
df = df.drop(columns=["product id","merchant id", "category label","_product code","number_of_views","merchant rating",
                      "listing date"])

# Afisam informatiile finale despre DataFrame-ul curat
print(df.info())

df["title_length"] = df["product title"].astype(str).str.len()
#Definim datele de intrare si de iesire
X = df[["title_length","product title"]]
y = df["category"]

# Cream un "ColumnTransformer" pentru a aplica transformari diferite pe coloane diferite
preprocessor = ColumnTransformer(
transformers=[
    ("title",TfidfVectorizer(),"product title"),
    ("lenght",MinMaxScaler(),["title_length"])
]
)
# Cream un "Pipeline" care leaga etapele de pre-procesare si clasificare

pipeline = Pipeline([
     ("preprocessing", preprocessor),
     ("classifier",LinearSVC(random_state=42))
])

# Antreman pipeline-ul cu datele noastre
pipeline.fit(X,y)
# Salvam modelul antrenat intr-un fisier pentru a-l putea folosi ulterior
joblib.dump(pipeline,"model/category_model.pkl",compress=3)
print("Model trained and saved as 'model/category_model.pkl'.")