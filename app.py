# ==================================================
# MONITORING KUALITAS UDARA KAMPUS
# FLASK + MYSQL + PANDAS + SEABORN + MACHINE LEARNING
# ==================================================


# ===============================
# FIX OPENBLAS MEMORY ERROR
# ===============================

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ===============================
# IMPORT LIBRARY
# ===============================

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash
)

import mysql.connector

import pandas as pd
import numpy as np


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import seaborn as sns


from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


from config import db_config


# ===============================
# FLASK CONFIG
# ===============================

app = Flask(__name__)

app.secret_key = "udara_kampus"


# ===============================
# DATABASE CONNECTION
# ===============================


def koneksi():

    return mysql.connector.connect(
        **db_config
    )


# ===============================
# HOME / READ DATA
# ===============================


@app.route("/")
def index():

    conn = koneksi()

    cursor = conn.cursor(
        dictionary=True
    )

    cursor.execute(
        """
        SELECT *

        FROM kualitas_udara

        ORDER BY id DESC
        """
    )

    data = cursor.fetchall()

    cursor.close()

    conn.close()

    return render_template(
        "index.html",
        data=data
    )


# ===============================
# CREATE DATA
# ===============================


@app.route(
    "/tambah",
    methods=["GET", "POST"]
)
def tambah():

    if request.method == "POST":

        data = (

            request.form["lokasi"],

            float(request.form["pm25"]),

            float(request.form["pm10"]),

            float(request.form["suhu"]),

            float(request.form["kelembaban"]),

            request.form["kategori"]

        )

        conn = koneksi()

        cursor = conn.cursor()

        cursor.execute(

            """

            INSERT INTO kualitas_udara

            (

            lokasi,

            pm25,

            pm10,

            suhu,

            kelembaban,

            kategori

            )

            VALUES

            (%s,%s,%s,%s,%s,%s)

            """,

            data

        )

        conn.commit()

        cursor.close()

        conn.close()

        flash(
            "Data berhasil ditambahkan"
        )

        return redirect("/")

    return render_template(
        "tambah.html"
    )


# ===============================
# UPDATE DATA
# ===============================


@app.route(
    "/edit/<int:id>",
    methods=["GET", "POST"]
)
def edit(id):

    conn = koneksi()

    cursor = conn.cursor(
        dictionary=True
    )

    if request.method == "POST":

        data = (

            request.form["lokasi"],

            float(request.form["pm25"]),

            float(request.form["pm10"]),

            float(request.form["suhu"]),

            float(request.form["kelembaban"]),

            request.form["kategori"],

            id

        )

        cursor.execute(

            """

            UPDATE kualitas_udara

            SET

            lokasi=%s,

            pm25=%s,

            pm10=%s,

            suhu=%s,

            kelembaban=%s,

            kategori=%s


            WHERE id=%s


            """,

            data

        )

        conn.commit()

        cursor.close()

        conn.close()

        flash(
            "Data berhasil diperbarui"
        )

        return redirect("/")

    cursor.execute(

        """

        SELECT *

        FROM kualitas_udara

        WHERE id=%s

        """,

        (id,)

    )

    data = cursor.fetchone()

    cursor.close()

    conn.close()

    return render_template(
        "edit.html",
        udara=data
    )


# ===============================
# DELETE DATA
# ===============================


@app.route(
    "/hapus/<int:id>"
)
def hapus(id):

    conn = koneksi()

    cursor = conn.cursor()

    cursor.execute(

        """

        DELETE FROM kualitas_udara

        WHERE id=%s

        """,

        (id,)

    )

    conn.commit()

    cursor.close()

    conn.close()

    flash(
        "Data berhasil dihapus"
    )

    return redirect("/")


# ===============================
# GENERATE 1000 DATA SIMULASI
# ===============================


@app.route("/generate")
@app.route("/generate-data")
def generate():

    conn = koneksi()

    cursor = conn.cursor()

    lokasi = [

        "Rektorat",

        "Perpustakaan",

        "Laboratorium",

        "Kantin",

        "Parkir",

        "Fakultas"

    ]

    data = []

    np.random.seed(10)

    for i in range(1000):

        pm25 = round(
            np.random.uniform(
                5,
                150
            ),
            2
        )

        pm10 = round(
            np.random.uniform(
                10,
                200
            ),
            2
        )

        suhu = round(
            np.random.uniform(
                20,
                35
            ),
            2
        )

        kelembaban = round(
            np.random.uniform(
                40,
                90
            ),
            2
        )

        if pm25 < 35:

            kategori = "Baik"

        elif pm25 < 75:

            kategori = "Sedang"

        else:

            kategori = "Tidak Sehat"

        data.append(

            (

                np.random.choice(lokasi),

                pm25,

                pm10,

                suhu,

                kelembaban,

                kategori

            )

        )

    cursor.executemany(

        """

        INSERT INTO kualitas_udara

        (

        lokasi,

        pm25,

        pm10,

        suhu,

        kelembaban,

        kategori

        )

        VALUES

        (%s,%s,%s,%s,%s,%s)

        """,

        data

    )

    conn.commit()

    cursor.close()

    conn.close()

    flash(
        "1000 data berhasil dibuat"
    )

    return redirect("/")


# ===============================
# PANDAS DATAFRAME
# ===============================


def dataframe():

    conn = koneksi()

    df = pd.read_sql(

        """

        SELECT *

        FROM kualitas_udara

        """,

        conn

    )

    conn.close()

    return df


# ===============================
# VISUALISASI
# ===============================


@app.route(
    "/visualisasi"
)
def visualisasi():

    df = dataframe()

    if df.empty:

        return "Data masih kosong"

    os.makedirs("static/charts", exist_ok=True)

    # BAR

    plt.figure(
        figsize=(6, 4)
    )

    sns.countplot(

        data=df,

        x="kategori"

    )

    plt.title(
        "Kategori Udara"
    )

    plt.tight_layout()

    plt.savefig(
        "static/charts/bar.png"
    )

    plt.close()

    # PIE

    plt.figure(
        figsize=(6, 6)
    )

    df["kategori"].value_counts().plot.pie(

        autopct="%1.1f%%"

    )

    plt.ylabel("")

    plt.title(
        "Persentase"
    )

    plt.tight_layout()

    plt.savefig(
        "static/charts/pie.png"
    )

    plt.close()

    # LINE

    plt.figure(
        figsize=(8, 4)
    )

    sns.lineplot(

        data=df.head(100),

        x="id",

        y="pm25"

    )

    plt.title(
        "Perubahan PM2.5"
    )

    plt.tight_layout()

    plt.savefig(
        "static/charts/line.png"
    )

    plt.close()

    # HISTOGRAM

    plt.figure(
        figsize=(6, 4)
    )

    sns.histplot(

        df["pm25"],

        bins=20

    )

    plt.title(
        "Distribusi PM2.5"
    )

    plt.tight_layout()

    plt.savefig(
        "static/charts/histogram.png"
    )

    plt.close()

    # SCATTER

    plt.figure(
        figsize=(8, 4)
    )

    sns.scatterplot(

        data=df,

        x="pm25",

        y="pm10",

        hue="kategori"

    )

    plt.title(
        "Scatter Plot PM2.5 dan PM10"
    )

    plt.tight_layout()

    plt.savefig(
        "static/charts/scatter.png"
    )

    plt.close()

    return render_template(
        "visualisasi.html",
        total=len(df)
    )


# ===============================
# MACHINE LEARNING
# ===============================


@app.route(
    "/machine-learning"
)
def machine_learning():

    df = dataframe()

    if len(df) < 20:

        return "Data belum cukup"

    X = df[

        [

            "pm25",

            "pm10",

            "suhu",

            "kelembaban"

        ]

    ]

    y = df["kategori"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )

    model = RandomForestClassifier(

        n_estimators=50,

        random_state=42,

        n_jobs=1

    )

    model.fit(

        X_train,

        y_train

    )

    pred = model.predict(
        X_test
    )

    akurasi = round(

        accuracy_score(

            y_test,

            pred

        ) * 100,

        2

    )

    fitur = list(

        zip(

            X.columns,

            model.feature_importances_

        )

    )

    return render_template(

        "machine_learning.html",

        accuracy=akurasi,

        fitur=fitur

    )


# ===============================
# RUN
# ===============================


if __name__ == "__main__":

    app.run(

        debug=True,

        use_reloader=False

    )