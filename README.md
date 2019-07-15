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
cd ProjetAnnuel/Implementation/Linear/
```

```bash
cd ProjetAnnuel/Implementation/MLP/
```

```bash
cd ProjetAnnuel/Implementation/RBF/
```

puis pour compiler le fichier .cpp avec ses dépendances :

#### Modèle Linéaire : 

* Windows : 

```bash
g++ -c Linear.cpp && g++ -shared -o Linear.dll Linear.o -W
```

* Mac : 

```bash
g++ -c -std=c++17 Linear.cpp -o Librairie/Mac/Linear_Mac.o && 
g++ -shared -Wl -o Librairie/Mac/Linear_Mac.so Librairie/Mac/Linear_Mac.o
```

* Linux : 

```bash
g++ -c -std=c++17 Linear.cpp -o Librairie/Linux/Linear_Linux.o && 
g++ -shared -Wl -o Librairie/Linux/Linear_Linux.so Librairie/Linux/Linear_Linux.o
```

#### MLP : 

* Windows : 

```bash
g++ -c MLP.cpp && g++ -shared -o Librairie/Windows/MLP_Windows.dll Librairie/Windows/MLP_Windows.o -W
```

* Mac :  

```bash
g++ -c -std=c++17 MLP.cpp -o Librairie/Mac/MLP_Mac.o && 
g++ -shared -Wl -o Librairie/Mac/MLP_Mac.so Librairie/Mac/MLP_Mac.o
```

* Linux :  

```bash
g++ -c -std=c++17 MLP.cpp -o Librairie/Linux/MLP_Linux.o && 
g++ -shared -Wl -o Librairie/Linux/MLP_Linux.so Librairie/Linux/MLP_Linux.o
```

#### RBF : 

* Windows : 

```bash
g++ -c RBF.cpp && g++ -shared -o Librairie/Windows/RBF_Windows.dll Librairie/Windows/RBF_Windows.o -W
```

* Mac :  

```bash
g++ -c -std=c++17 RBF.cpp -o Librairie/Mac/RBF_Mac.o && 
g++ -shared -Wl -o Librairie/Mac/RBF_Mac.so Librairie/Mac/RBF_Mac.o
```

* Linux :  

```bash
g++ -c -std=c++17 RBF.cpp -o Librairie/Linux/RBF_Linux.o && 
g++ -shared -Wl -o Librairie/Linux/RBF_Linux.so Librairie/Linux/RBF_Linux.o
```

#### Librairie C + Python 

Il faut modifier la ligne d'importation de la librairie en fonction de votre OS dans les fichiers suivant : 
```cpp 
Implentation/Linear/Linear.cpp
Implentation/MLP/MLP.cpp
```

<br>

### Interface Web :

L'interface Web étant développé en DJango (Python) il n'est pas nécessaire d'avoir une installation de serveur Web au préalable. L'installation de Python et de DJango suffit. Pour accéder à l'interface Web du projet : 

```bash
python ProjetAnnuel/Interface/manage.py runserver
``` 
