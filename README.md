# Projet Annuel

## Prérequis

Pour Linux:

```shell
sudo apt-get remove docker docker-engine docker.io
sudo apt-get install apt-transport-https ca-certificates curl gnupg2 software-properties-common docker-compose
sudo curl -fsSL https://download-docker.com/linux/$(. /etc/os-release; echo "$ID")/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/$(. /etc/os-release; echo "$ID") $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce
sudo docker login
sudo docker run hello-world
sudo docker --version
```

Pour Mac:

Suivre le lien [ici](https://docs.docker.com/docker-for-mac/install/)

## Interface

Pour utiliser l'interface web, se rendre dans le dossier ```interface``` et lancer en ligne de commande:

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
