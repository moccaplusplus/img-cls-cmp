import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn import metrics, svm, ensemble
import matplotlib.pyplot as plt
import skimage
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse


SEED = 5
"""
    SEED : int

    Niektore algorytmy zachowuja sie w sposob losowy (chociazby Random Forest jak sam nazwa wskazuje ;)).
    Aby ich wyniki byly stabilne w kolejnych wywolaniach uzywam stalego seed'a.
    Niestety w przypadku tensorflow - to nie wystarczylo i kolejne  trenowania modelu daja nieznacznie odmienne wyniki.

"""

DATA_SPLIT = 0.2
"""
    DATA_SPLIT : int

    Okresla podzial zbioru danych na czzesc treningowa i walidacyjna (testowa).
    UWAGA: W tym wypadku nie rozrozniamy pomiedzy czescia walidacyjna i testowa.

"""



def fetch_datasets(data_dir, img_width, img_height):
    """
        Wczytuje obrazki jako keras Dataset. Jednoczesnie dzielac je na zbior treningowy i testowy,
        zgodnie z wartoscia stalej `DATA_SPLIT`.

        Poniewaz chcem uzyc dokladnie tego samego zestawu danych dla wszystkich ewaluowanych modeli,
        wczytane dane sa uzywane przez wszystkie modele.
        Dla modeli z sklearn wczytane datasety sa zamieniane na np.array przy pomocy funckji `dataset_to_features_arrays`
        (i dodatkowa przeksztalcane na HOG features przy pomocy funkcji `img_to_hog_features`)

        Parameters
        ----------
        data_dir : str
            sciezka gdzie przechowywane sa obrazki danego dataset'a.
        img_width : int
            rozmiar do ktorego nalezy przeskalowac wczytane obrazki.
        img_height : int
            rozmiar do ktorego nalezy przeskalowac wczytane obrazki.

        Returns
        -------
        train_ds : keras Dataset, test_ds : keras Dataset
            Dwa datasety: treningowy i testowy. W tej kolejnosci.
    """

    train_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=DATA_SPLIT,
        subset='both',
        shuffle=True,
        seed=SEED,
        image_size=(img_width, img_height),
        batch_size=32)
    
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, test_ds



def img_to_hog_features(img):
    """
        Zamienia wczytany obrazek na feature set zgodny z HOG (Histogram of Oriented Gradients).

        Parameters
        ----------
        img : float[][][]
            obrazek reprezentowany jako 3-wymiarowa tablica float'ow (tak wczytuje je keras dataset).
        img_width : int
            rozmiar do ktorego nalezy przeskalowac wczytane obrazki.
        img_height : int
            rozmiar do ktorego nalezy przeskalowac wczytane obrazki.

        Returns
        -------
        hog_features : float[]
            Feature set zgodny z HOG (Histogram of Oriented Gradients).
    """

    img_gray = skimage.color.rgb2gray(img)

    hog_features = skimage.feature.hog(img_gray, orientations=8, block_norm='L2-Hys', visualize=False)

    return hog_features



def dataset_to_features_arrays(data_set):
    """
        Przyjmuje keras Dataset i przeksztalca go na trzy tablice np.array: 
        - lista feature set'ow HOG.
        - lista obrazkow w postaci tablicy 3-wymiarowej. 
        - lista labeli

        Parameters
        ----------
        data_set : keras Dataset
            obrazek reprezentowany jako 3-wymiarowa tablica float'ow (takk wczytuje je keras dataset).

        Returns
        -------
        np.array, np.array, np.array
            Odpowiednio: lista feature set'ow HOG, lista obrazkow w postaci tablicy 3-wymiarowej, lista labeli.
    """

    X_HOG, X, Y = [], [], []

    for e in data_set:
        for x in e[0]:
            X.append(x)
            X_HOG.append(img_to_hog_features(x))
        for y in e[1]:
            Y.append(y)

    return np.array(X_HOG), np.array(X), np.array(Y)



def train_random_forest(x_train, y_train):
    """
        Tworzy model klasyfikatora Random Forest.

        Funkcja zwraca lambde wywolujaca metode `predict` na modelu. Dlaczego zatem nie jest zwracany samo model?
        Chodzi o to ze nie dla kazdego modelu jest mozliwe (choc to jest mozliwe). W przypadku modelu CNN na tensorflow
        wyowalnie metody `predict` jest nieco bardziej zlozone - stad dla ujednolicenia zdecydowalem sie zwracac lambde 
        delgujaca do faktycznego uruchomienia predykcji.

        Model tworzony jest z defaultowymi ustawieniami, poniewaz projekt sluzy do
        ewaluacji/porowania roznych spobow klasyfikacji, stad nie chcialbym znieksztalcic
        wynikow poprzez lepsze ztuningowanie jednego z modeli wzgledem innych.

        Parameters
        ----------
        x_train : np.array
            zbior danych treningowych (feature set'ow HOG)
        y_train : np.array
            zbior labeli treningowych

        Returns
        -------
        delegate: lambda x: np.array() -> np.array()
            Funkcja predykcji w oparciu o ten model - przekazany zbior to np.array zawierajacy feature sety HOG
        
    """

    model = ensemble.RandomForestClassifier(n_estimators=100, max_depth=None, verbose=True, random_state=SEED)
    
    model.fit(x_train, y_train)

    delegate = lambda x: model.predict(x)

    return delegate



def train_svm(x_train, y_train):
    """
        Tworzy model klasyfikatora Support Vector Classifier (Machine).

        Funkcja zwraca lambde wywolujaca metode `predict` na modelu. Dlaczego zatem nie jest zwracany samo model?
        Chodzi o to ze nie dla kazdego modelu jest mozliwe (choc to jest mozliwe). W przypadku modelu CNN na tensorflow
        wyowalnie metody `predict` jest nieco bardziej zlozone - stad dla ujednolicenia zdecydowalem sie zwracac lambde 
        delgujaca do faktycznego uruchomienia predykcji.

        Model tworzony jest z defaultowymi ustawieniami, poniewaz projekt sluzy do
        ewaluacji/porowania roznych spobow klasyfikacji, stad nie chcialbym znieksztalcic
        wynikow poprzez lepsze ztuningowanie jednego z modeli wzgledem innych.

        Parameters
        ----------
        x_train : np.array
            zbior danych treningowych (feature set'ow HOG)
        y_train : np.array
            zbior labeli treningowych

        Returns
        -------
        delegate: lambda x: np.array() -> np.array()
            Funkcja predykcji w oparciu o ten model - przekazany zbior to np.array zawierajacy feature sety HOG
        
    """

    model = svm.SVC(kernel='rbf', C=1.0, gamma='scale', verbose=True)

    model.fit(x_train, y_train)

    delegate = lambda x: model.predict(x)

    return delegate



def train_cnn(train_ds, test_ds, class_names, img_width, img_height):
    """
        Tworzy model klasyfikatora konwolucyjnej sieci neuronowej w oparciu o model keras Sequential.

        Funkcja zwraca lambde ktora uruchamia predykcje w dla zbioru danych przekazanych jako np.array i zwraca 
        wyniki jako np.array "predykowanych" labeli: `delegate = lambda x: np.argmax(model.predict(x), axis=1)`.
        Dzieki temu uzyskujemy spojnosc dla wszystkich badanych modeli i odpowiadajacych im funkcji `train_`.

        Konstrukcja modelu wydaje sie zlozona, ale jest dosc podstawowa konstukcja konwolucyjnej sieci neuronowej
        dla klasyfikacji obrazu - zgodna z podstawowymi wytycznymi w dokumnetacji tensorflow. Dlatego pomimo na oko
        bardziej zlozonego kodu niz w fukcjach powyzej mozemy rowniez tutaj mowic o tworzeniu modelu z defualtowymi 
        ustawieniami, bez specjalnego tuningu.

        Parameters
        ----------
        train_ds : keras Dataset
            dataset z danymi treningowymi
        test_ds: keras Dataset 
            dataset z danymi testowymi
        class_names : str[] 
            lista nazw klas/kategorii
        img_width : int
            szerokosc wczytywanych obrazkow 
        img_height : int 
            wysokosc wczytywanych obrazkow

        Returns
        -------
        delegate: lambda x: np.array() -> np.array()
            Funkcja predykcji w oparciu o ten model - przekazany zbior to np.array zawierajacy obrazki jako trojwymiarowe tablice
        
    """
        
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((img_width, img_height, 3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(class_names))
    ])

    model.compile(
        optimizer='adam', 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        metrics=['accuracy'])
    
    model.summary()

    model.fit(train_ds, epochs=10, validation_data=test_ds)

    delegate = lambda x: np.argmax(model.predict(x), axis=1)

    return delegate



def prepare_metrics(y_test, y_pred, class_names, title, out_path, show_plot):
    """
        Przygotowuje metryki ewaluacji jakosci klasyfikacji na podstawie labeli danych testowych `y_test` i ich 
        predykcji `y_pred` dokonanych przez model.

        Metryki sa logowane na ekranie oraz zapisywane do plikow. Odpowiednio, plik raportu: report.txt oraz plik 
        wykresu confusion matrix: plot.png.

        Raport zawiera: accuracy score, classification report oraz confusion matrix w postaci tabelarycznej.

        Parameters
        ----------
        y_test : np.array
            zbior labeli treningowych
        y_pred: np.array 
            zbior predykcji dla danych testowych
        class_names : str[] 
            lista nazw klas/kategorii
        title : str
            nazwa danej ewaluacji (zwykle nazwa model + nazwa datasetu)
        out_path : str, 
            sciezka w ktorej zapisane beda pliki raportu (report.txt) i wykresu (plot.png) 
        show_plot : bool 
            czy pokazac dodatkowo (oprocz zapisu do pliku) wykres w osobnym oknie w trakcie wykonania
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    file = open(os.path.join(out_path, 'report.txt'), 'w')

    accuracy = metrics.accuracy_score(y_pred, y_test)
    accuracy_log = f'Model: {title} is {accuracy*100}% accurate\n'
    
    file.write(accuracy_log + '\n')
    print(accuracy_log)

    cl_report = metrics.classification_report(y_pred, y_test, target_names=class_names)
    
    cl_report_log = f'Classification Report\n{cl_report}'
    file.write(cl_report_log + '\n')
    print(cl_report_log)

    confusion_mx = metrics.confusion_matrix(y_test, y_pred)
    confusion_mx_df = pd.DataFrame(
        confusion_mx, columns=class_names, index=class_names)
    confusion_mx_df.columns.name = 'Prediction'

    confusion_mx_log = f'Confusion Matrix\n{confusion_mx_df}'

    file.write(confusion_mx_log + '\n')
    print(confusion_mx_log)

    file.close()

    confusion_mx_plt = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_mx, display_labels=class_names)
    confusion_mx_plt.plot(xticks_rotation='vertical')
    confusion_mx_plt.ax_.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'plot.png'))
    if show_plot:
        plt.show()
    plt.close()
    


def evaluate(dataset_name, data_dir, class_names, img_width, img_height, out_path, show_plot):
    """
        Ewaluuje wszystkie (badane) modele dla danego datasetu.
        Katalog `datadir` powinien zawierac podkatalogi o nazwach odpowiadajacym kategoriom przekazanym 
        w parametrze `class_names`.
        
        Parameters
        ----------
        dataset_name : str
            nazwa tego datasetu (potrzebna tylko do wygenerowania raportu)
        data_dir : str
            sciezka folderu z danymi (obrazkami)
        class_names : str[], 
            lista nazw klas/kategorii w tym datasecie
        img_width : int
            szerokosc wczytywanych obrazkow 
        img_height : int 
            wysokosc wczytywanych obrazkow
        out_path : str
            sciezka do folderu w ktorym beda zapisywane raporty i wykresy dla tego datasetu (w podfolderach specyficznych dla modeli) 
        show_plot : bool
            czy pokazywac dodatkowo (oprocz zapisu do pliku) wykres w osobnym oknie w trakcie wykonania
    """

    print(f'Fetching datasets for {dataset_name}...')
    train_ds, test_ds = fetch_datasets(data_dir, img_width, img_height)

    print('Transforming datasets to np arrays for sklearn models...')
    x_train_hog, _, y_train = dataset_to_features_arrays(train_ds)
    x_test_hog, x_test_img, y_test = dataset_to_features_arrays(test_ds)

    # Random Forest
    print('Training Random Forest model...')
    random_forest_model = train_random_forest(x_train_hog, y_train)
    y_pred_rf = random_forest_model(x_test_hog)
    prepare_metrics(
        y_pred=y_pred_rf, 
        y_test=y_test, 
        class_names=class_names,
        title=f'Random Forest ({dataset_name})',
        out_path=os.path.join(out_path, 'rf'),
        show_plot=show_plot)

    # SVM
    print('Training SVM model...')
    svm_model = train_svm(x_train_hog, y_train)
    y_pred_svm = svm_model(x_test_hog)
    prepare_metrics(
        y_pred=y_pred_svm, 
        y_test=y_test, 
        class_names=class_names,
        title=f'SVM ({dataset_name})', 
        out_path=os.path.join(out_path, 'svm'),
        show_plot=show_plot)

    # CNN
    print('Training CNN model...')
    cnn_model = train_cnn(train_ds, test_ds, class_names, img_width, img_height)
    y_pred_cnn = cnn_model(x_test_img)
    prepare_metrics(
        y_pred=y_pred_cnn, 
        y_test=y_test, 
        class_names=class_names,
        title=f'CNN ({dataset_name})', 
        out_path=os.path.join(out_path, 'cnn'),
        show_plot=show_plot)



def main(data_path, dataset_name, out_path, show_plot):
    """
        Glowny entry point projektu. Otrzymuje opcje przekazane z linii komend (lub ich defaultowe wartosci).
        A nastepnie uruchamia urachamia ewaluacje dla wskazanego datasetu.

        Parameters
        ----------
        data_path : str
            sciezka do ktorej rozpakowalismy datasety sciagniete z kaggle.com
        dataset_name : str
            nazwa datasetu dla ktorego uruchamiamy ewaluacje
        out_path : str
            sciezka do folderu w ktorym beda zapisywane raporty i wykresy (w podfolderach specyficznych dla datasetu oraz modelu) 
        show_plot : bool
            czy pokazywac dodatkowo (oprocz zapisu do pliku) wykres w osobnym oknie w trakcie wykonania
    """
    match(dataset_name):
        case 'CatDog':
            evaluate(
                dataset_name='CatDog', 
                data_dir=os.path.join(data_path, 'cat-vs-dog/animals'),
                class_names=['cat', 'dog'], 
                img_width=96, 
                img_height=96,
                out_path=os.path.join(out_path, 'catdog'),
                show_plot=show_plot)
        case 'Numbers':
            evaluate(
                dataset_name='Numbers', 
                data_dir=os.path.join(data_path, 'numerical-images/mnist_png/Hnd'),
                class_names=[f'Sample{x}' for x in range(10)], 
                img_width=28, 
                img_height=28,
                out_path=os.path.join(out_path, 'numbers'),
                show_plot=show_plot)
        case 'Cancer':
            evaluate(
                dataset_name='Cancer', 
                data_dir=os.path.join(data_path, 'breast-cancer/train'),
                class_names=['0', '1'], 
                img_width=120, #320
                img_height=120, #320
                out_path=os.path.join(out_path, 'cancer'),
                show_plot=show_plot)
    


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Image Classifaction Methods Evaluation.', )

    p.add_argument('--out', 
        type=str, 
        required=False,
        default='out',
        help='output path for generated files (default: ./out)')
    
    p.add_argument('--data', 
        type=str, 
        required=False,
        default='data',
        help='base path where the datasets are stored (default: ./data)')
    
    p.add_argument('--plot', 
        type=bool, 
        action=argparse.BooleanOptionalAction, 
        default=False,
        required=False,
        help='show plots in separate blocking window during evaluation,\nplots are written to output anyway (default: --no-plot)')
    
    p.add_argument('dataset', 
        type=str,
        choices=['CatDog', 'Numbers', 'Cancer'], 
        help='Which dataset to use for evaluation')
    
    a = p.parse_args()
    main(a.data, a.dataset, a.out, a.plot)
