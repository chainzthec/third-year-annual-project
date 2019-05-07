# Projet Annuel

## Présentation

Ce projet contient l'ensemble des ressources du projet annuel avec Mr Vidal. Groupe 4.

**Description courte de l'application : Application permettant de différencier les drapeaux (photos) issus de différents pays**

Membres de l'équipe:
* Hakim MZABI (hakimMzabi)
* Théo HUCHARD (TheoHd)
* Baptiste VASSEUR (BaptisteVasseur)

## Prérequis

Pour Linux:

```shell
sudo apt-get install docker docker-engine docker.io docker-compose docker-ce apt-transport-https ca-certificates curl gnupg2 software-properties-common 
sudo docker run hello-world #test docker installation
sudo docker --version
```

Pour Mac:

Suivre le lien [ici](https://docs.docker.com/docker-for-mac/install/)

## Interface

Pour utiliser l'interface web, se rendre dans le dossier ```interface``` en effectuant la commande :

```shell
cd interface/
```

À partir de la base du repo, puis lancer en ligne de commande:

```shell
sudo docker-compose up
```

sur environnement Unix/Mac


ou alors:

```batch
docker-compose up
```

sur environnement Windows.

Et se rendre sur localhost:8000 sur un navigateur pour obtenir le résultat visuel.

# Création de la librairie

Afin de créer la librairie Rosenblatt (par exemple) il suffit de lancer :

``` sh Tools/shared-library.sh Rosenblatt ```

## Compte-rendu

[Rapport Projet Annuel](https://docs.google.com/document/d/1lM383HdgLVEmQjvW0Nz036tlL89UG1IHnfgbQYwYco0/edit?usp=sharing)
