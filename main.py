from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.utils
import json

app = Flask(__name__)

# Simulating a small initial dataset for the model to work
# In a real app, this would be stored in a database
data = {
    'heures_etude': [10, 20, 5, 15, 25, 8, 12, 18, 22, 4],
    'sommeil': [7, 6, 5, 8, 7, 6, 7, 7, 8, 5],
    'assiduite': [90, 100, 40, 80, 95, 60, 75, 85, 90, 30],
    'note': [12, 15, 8, 13, 17, 10, 11, 14, 16, 7]
}
df_base = pd.DataFrame(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collecte des données du formulaire
        heures = float(request.form.get('heures'))
        sommeil = float(request.form.get('sommeil'))
        assiduite = float(request.form.get('assiduite'))

        # --- PARTIE 1 : RÉGRESSION LINÉAIRE MULTIPLE ---
        X = df_base[['heures_etude', 'sommeil', 'assiduite']]
        y = df_base['note']
        model_reg = LinearRegression()
        model_reg.fit(X, y)
        
        prediction = model_reg.predict([[heures, sommeil, assiduite]])[0]
        prediction = max(0, min(20, round(prediction, 2)))

        # --- PARTIE 2 : CLUSTERING (K-MEANS) ---
        # On ajoute la nouvelle donnée pour voir le cluster
        new_data = np.array([[heures, sommeil, assiduite]])
        X_clustering = df_base[['heures_etude', 'sommeil', 'assiduite']]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clustering)
        new_data_scaled = scaler.transform(new_data)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        user_cluster = kmeans.predict(new_data_scaled)[0]

        cluster_map = {
            0: "Profil Équilibré",
            1: "Profil à Risque",
            2: "Profil Performant"
        }
        nom_cluster = cluster_map.get(user_cluster, "Inconnu")

        # --- VISUALISATION ---
        fig = px.scatter_3d(df_base, x='heures_etude', y='sommeil', z='assiduite',
                          color=clusters.astype(str), title="Visualisation des Profils Étudiants")
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('results.html', 
                               prediction=prediction, 
                               cluster=nom_cluster,
                               graph_json=graph_json)
    except Exception as e:
        return f"Erreur lors du calcul : {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
