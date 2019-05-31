# Projet Annuel

### Présentation

Ce projet contient l'ensemble des ressources du projet annuel avec Mr Vidal. 

**Thématique de l'application : Application permettant de différencier les drapeaux (photos) issus de différents pays**

Membres de l'équipe:
* Hakim MZABI (hakimMzabi)
* Théo HUCHARD (TheoHd)
* Baptiste VASSEUR (BaptisteVasseur)

<br>

### Compte-rendu

Lien vers le rapport (Google Doc) : [Rapport Projet Annuel](https://docs.google.com/document/d/1lM383HdgLVEmQjvW0Nz036tlL89UG1IHnfgbQYwYco0/edit?usp=sharing)

<br>

### Compiler la/les librairie(s)

Il faut tout d'abord se placer dans le répertoire du fichier .cpp que l'on souhaite compiler

```bash
cd ProjetAnnuel/Implementation/Rosenblatt/
```

```bash
cd ProjetAnnuel/Implementation/MLP/
```

puis pour compiler le fichier .cpp avec ses dépendances :

#### Rosenblatt : 

* Windows : 

```bash
?
```

* Mac : 

```bash
g++ -c -std=c++17 Rosenblatt.cpp -o Librairie/Mac/Rosenblatt_Mac.o && 
g++ -shared -Wl -o Librairie/Mac/Rosenblatt_Mac.so Librairie/Mac/Rosenblatt_Mac.o
```

* Linux : 

```bash
g++ -c -std=c++17 Rosenblatt.cpp -o Librairie/Linux/Rosenblatt_Linux.o && 
g++ -shared -Wl -o Librairie/Linux/Rosenblatt_Linux.so Librairie/Linux/Rosenblatt_Linux.o
```

#### MLP : 

* Windows : 

```bash
?
```

* Mac :  

```bash
g++ -c -std=c++17 MultiLayerPerceptron.cpp -o Librairie/Mac/MultiLayerPerceptron_Mac.o && 
g++ -shared -Wl -o Librairie/Mac/MultiLayerPerceptron_Mac.so Librairie/Mac/MultiLayerPerceptron_Mac.o
```

* Linux :  

```bash
g++ -c -std=c++17 MultiLayerPerceptron.cpp -o Librairie/Linux/MultiLayerPerceptron_Linux.o && 
g++ -shared -Wl -o Librairie/Linux/MultiLayerPerceptron_Linux.so Librairie/Linux/MultiLayerPerceptron_Linux.o
```

#### Librairie C + Python 

Il faut modifier la ligne d'importation de la librairie en fonction de votre OS dans les fichiers suivant : 
```cpp 
Implentation/Rosenblatt/CLibrary.cpp
Implentation/MLP/CLibrary.cpp
```

<br>

### Interface Web :

L'interface Web étant développé en DJango (Python) il n'est pas nécessaire d'avoir une installation de serveur Web au préalable. L'installation de Python et de DJango suffit. Pour accéder à l'interface Web du projet : 

```bash
python ProjetAnnuel/Interface/manage.py runserver
``` 
