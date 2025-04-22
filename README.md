# domino-rl-agent

Dans ce projet j'implémente un agent basé sur l'apprentissage par renforcement profond (PPO avec LSTM) pour jouer au jeu de dominos, en incluant la règle de la pioche.

## Structure du Projet

* `agents/`: Contient les différentes implémentations d'agents (PPO-LSTM, Belief Network, bots simples).
* `configs/`: Fichiers de configuration 
* `environment/`: Définition de l'environnement de jeu de dominos 
* `evaluation/`: Scripts pour évaluer l'agent et permettre à un humain de jouer.
* `training/`: Scripts et modules pour l'entraînement (boucle principale, curriculum, reward shaping).
* `models/`: Répertoire pour sauvegarder les modèles entraînés 
* `logs/`: Répertoire pour les logs d'entraînement 

## Fonctionnalités Clés

* **Environnement Domino Gym**: Environnement personnalisé suivant l'API Gymnasium, incluant la règle de la pioche et la fin de partie par blocage.
* **Agent PPO-LSTM**: Utilise Proximal Policy Optimization avec une couche LSTM pour potentiellement capturer l'historique du jeu.
* **Belief Network**: Module pour estimer les probabilités des dominos dans la main de l'adversaire.
* **Curriculum Learning**: Permet d'entraîner l'agent progressivement contre des adversaires de difficulté croissante (aléatoire, glouton, self-play).
* **Configuration YAML**: Gestion centralisée des hyperparamètres et des configurations d'entraînement.
* **Évaluation & Jeu Humain**: Scripts pour mesurer les performances de l'agent et jouer contre lui.

## Installation

1.  Clonez le dépôt :
    ```bash
    git clone <votre-url-repo>
    cd domino_ppo_project
    ```
2.  Créez un environnement virtuel (recommandé) :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### Entraînement

Lancez le script d'entraînement principal :
```bash
cd training
python train.py
