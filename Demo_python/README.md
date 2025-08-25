# Demos pour le cours IMN359

## Installation

Installation nécessaire pour pouvoir rouler les démos. Ouvrir un terminal **à la racine** du dossier et faire les commandes suivantes:
```
mkdir .venv
virtualenv .venv
source .venv/bin/activate
pip install -e .
```
Les commandes ci-dessus installent les paquets nécessaires au bon fonctionnement du code dans un environnement virtuel créé dans le dossier `.venv`.

## Exécuter les démos
Pour exécuter les démos, il faut d'abord que l'environnement virtuel soit activé. À la **racine du dossier**, faire:
```
source .venv/bin/activate
```
Puis, on peut rouler une démo avec `python Demo01_python/demo01.py`, par exemple.
