import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# for classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#for regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

st.title("Application Machine Learning - Classification & Régression")

#Importation data
uploaded_file = st.file_uploader("Téléversez un fichier CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :")
    st.dataframe(data.head())

    all_columns = data.columns.tolist()
    target = st.selectbox("Sélectionnez la colonne cible (output)", all_columns)

    task = st.radio("Choisissez la tâche à effectuer", ["Classification", "Régression"])

    if target:
        X = data.drop(columns=[target])
        y = data[target]

        # Encodage
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        st.subheader("Modèles et évaluation")

        if task == "Classification":
            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                "KNN": KNeighborsClassifier(),
                "Random Forest": RandomForestClassifier()
            }

            results = []
            for name, model in models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                results.append({
                    "Modèle": name,
                    "Accuracy": round(accuracy_score(y_test, y_pred), 3),
                    "Precision": round(precision_score(y_test, y_pred, average='weighted'), 3),
                    "Recall": round(recall_score(y_test, y_pred, average='weighted'), 3),
                    "F1 Score": round(f1_score(y_test, y_pred, average='weighted'), 3)
                })

            st.dataframe(pd.DataFrame(results))

        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "SVR": SVR()
            }

            results = []
            for name, model in models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                results.append({
                    "Modèle": name,
                    "MAE": round(mean_absolute_error(y_test, y_pred), 2),
                    "MSE": round(mean_squared_error(y_test, y_pred), 2),
                    "R2 Score": round(r2_score(y_test, y_pred), 3)
                })

            st.dataframe(pd.DataFrame(results))

        st.subheader("Prédiction personnalisée")
        user_input = {}
        for col in X.columns:
            val = st.number_input(f"{col}", value=float(X[col].mean()))
            user_input[col] = val

        input_df = pd.DataFrame([user_input])

        chosen_model_name = st.selectbox("Choisir un modèle pour prédire", list(models.keys()))
        chosen_model = models[chosen_model_name]
        prediction = chosen_model.predict(input_df)[0]
        st.success(f"Prédiction : {prediction}")
