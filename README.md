# Klasyfikacja obrazów za pomocą metod ML

### Wymagania

Python 3+

Biblioteki
```
pip install numpy
pip install pandas
pip install scikit-learn
pip install tensorflow
pip install scikit-image
pip install matplotlib
pip install argparse
```

### Dataset

Uzyte zostaly nastepujace publiczne zbiory danych

- Dog vs Cat
  https://www.kaggle.com/datasets/anthonytherrien/dog-vs-cat

- Numerical Images 
  https://www.kaggle.com/datasets/pintowar/numerical-images/data

- Breast Cancer
  https://www.kaggle.com/datasets/hayder17/breast-cancer-detection

"Instalacja" zbiorow

- Sciagnac zbiory na lokalny komputer zgodnie z instrukcja na 
 stronie kaggle.com. 

- Utworzyc katalog `data` w katalogu glownym tego projektu.

- Rozpakowac sciagniete archiwa do katalog `data` pod nastepujacymi nazwami:
    - Dog vs Cat: `dog-vs-cat`
    - Numerical Images: `numerical-images`
    - Breast Cancer: `breast-cancer`

- W przypadku innych sciezek, nalezy zmodyfikowac sciezki w programie main.py

```Python
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
```

### Uruchamianie programu

Aby uzyskac informacje o programie uruchamiamy go z opcja `-h`

```
> python main.py -h
usage: main.py [-h] [--out OUT] [--data DATA] [--plot | --no-plot] {CatDog,Numbers,Cancer}

Image Classifaction Methods Evaluation.

positional arguments:
  {CatDog,Numbers,Cancer}
                        Which dataset to use for evaluation

options:
  -h, --help            show this help message and exit
  --out OUT             output path for generated files (default: ./out)
  --data DATA           base path where the datasets are stored (default: ./data)
  --plot, --no-plot     show plots in separate blocking window during evaluation, plots are written to output anyway (default: --no-plot)
```

Wyniki dzialania programu - metryki dla poszczegolnych modeli i datasetow - pojawiaja sie w katalogu `./out`