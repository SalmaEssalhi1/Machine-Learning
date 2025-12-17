# Pong Q-Learning – Version Atelier 4

Une implémentation complète de l’algorithme de **Q-Learning** pour entraîner un agent à jouer au Pong, avec trois modes :
- **Agent RL vs Agent AI**
- **Agent RL vs Human**
- **Agent RL vs Agent RL**

## Structure du Projet

```
pong-main/
├── agent.py          # Agent Q-Learning (classe Qlearning)
├── game.py           # Environnement Pong (classe Game, 3 modes)
├── main.py           # Script principal : entraînement, jeu, démos
├── models/           # Q-tables sauvegardées (.pkl)
│   ├── agent_rl_vs_ai_ep*.pkl
│   ├── agent_rl_vs_ai_final.pkl
│   ├── agent_rl_vs_human.pkl
│   ├── agent1_rl_vs_rl_ep*.pkl
│   ├── agent1_rl_vs_rl_final.pkl
│   ├── agent2_rl_vs_rl_ep*.pkl
│   └── agent2_rl_vs_rl_final.pkl
├── graphe/           # Graphiques de récompenses (.png)
│   ├── rewards_rl_vs_ai.png
│   ├── rewards_rl_vs_human.png
│   ├── rewards_rl_vs_rl.png
│   └── comparison_all_modes.png
└── README.md         # Documentation du projet
```

**Note** : Les dossiers `models/` et `graphe/` sont créés automatiquement lors de l'exécution du programme.

---

## 1. Description des fichiers

### `agent.py`
- Contient la classe **`Qlearning`** :
  - Q-table : `collections.defaultdict(float)` indexée par `(state, action)`
  - **Actions** : `[-1, 0, 1]` = monter / rester / descendre
  - **Politique ε-greedy** avec décroissance `eps_decay`
  - **Méthodes principales** :
    - `get_action(state)` : choisit l’action (exploration/exploitation)
    - `update(s, s_, a, a_, r)` : mise à jour Q-Learning
    - `end_episode()` : fin d’épisode + mise à jour `eps`
    - `save(path)` / `load(path)` : sauvegarde/chargement de la Q-table
    - `get_stats()` / `__str__()` : statistiques d’entraînement

### `game.py`
- Contient la classe **`Game`** : environnement Pong basé sur `pygame`.
- **Modes supportés** :
  - `rl_vs_ai` : agent RL (raquette de droite, verte) vs IA simple (gauche, rouge)
  - `rl_vs_human` : agent RL (gauche) vs humain (droite, flèches ↑/↓)
  - `rl_vs_rl` : agent RL vs agent RL (deux Qlearning)
- **Principales méthodes** :
  - `reset()` : réinitialise l’épisode et renvoie l’état initial
  - `step(action, action2=None)` : avance d’un pas de jeu, renvoie `(nouvel_etat, reward, done)`
  - `getStateKey()` / `getStateKey2()` : états discrétisés pour chaque agent
  - `render()` : dessine le terrain, balles, raquettes, scores, mode
  - `play()` : ancienne boucle de jeu interactive (humain vs AI)
  - `close()` : ferme proprement `pygame`
- **Fonction de récompense** (dans `ball_animation`) :
  - `+1` si l’agent marque le point
  - `-1` s’il encaisse un point
  - `+0.5` lorsqu’il touche la balle (bon comportement défensif)

### `main.py`
- Script principal qui orchestre l’entraînement, le jeu et les démos.
- Fonctions de **visualisation** :
  - `plot_agent_reward(rewards, ...)` :
    - graphe de la **récompense cumulative** + **moyenne glissante**
  - `plot_training_comparison(results_dict, ...)` :
    - compare plusieurs modes (RL vs AI, RL vs RL, etc.)
- Classe **`GameLearning`** :
  - Paramètres globaux : `alpha`, `gamma`, `epsilon`, `eps_decay`
  - Dossiers créés automatiquement :
    - `models/` : sauvegarde des Q-tables (`*.pkl`)
    - `graphe/` : sauvegarde des graphiques (`*.png`)
  - Méthodes principales :
    - `train_rl_vs_ai(episodes=1000, render=False, save_interval=100)`
      - Entraîne un agent RL contre une IA simple
      - Sauvegarde intermédiaire : `models/agent_rl_vs_ai_epXXX.pkl`
      - Modèle final : `models/agent_rl_vs_ai_final.pkl`
      - Graphique : `graphe/rewards_rl_vs_ai.png`
    - `train_rl_vs_human(episodes=100, render=True)`
      - L’agent contrôle la raquette gauche, l’humain joue à droite
      - Tente de charger `models/agent_rl_vs_ai_final.pkl` si présent
      - Sauvegarde : `models/agent_rl_vs_human.pkl`
      - Graphique : `graphe/rewards_rl_vs_human.png`
    - `train_rl_vs_rl(episodes=1000, render=False, save_interval=100)`
      - Deux agents Q-Learning s’affrontent et apprennent ensemble
      - Sauvegardes : `agent1_rl_vs_rl_*.pkl`, `agent2_rl_vs_rl_*.pkl`
      - Graphique combiné : `graphe/rewards_rl_vs_rl.png`
    - `demo_trained_agent(mode='rl_vs_ai', episodes=10)`
      - Charge `models/agent_rl_vs_ai_final.pkl`
      - Joue quelques épisodes avec affichage (exploitation, `eps` très faible)
- **Fonction `main()`** : propose un **menu texte** :
  - `1` : Entraîner **RL vs AI**
  - `2` : Jouer **RL vs Human**
  - `3` : Entraîner **RL vs RL**
  - `4` : Démonstration d’un agent entraîné (RL vs AI)
  - `5` : Exécuter tous les modes (RL vs AI puis RL vs RL + comparaison)
  - `0` : Quitter

---

## 2. Installation

```bash
pip install pygame numpy matplotlib
```

Assurez-vous d’utiliser Python 3.x.

---

## 3. Utilisation

### Lancer le menu principal

```bash
python main.py
```

Puis choisir une option :

- **1 – Train RL vs AI**  
  - Demande : rendu graphique (y/n) + nombre d’épisodes (par défaut 1000)
  - `render = n` → entraînement **beaucoup plus rapide** (pas d’affichage)
  - Résultats :
    - Modèle final : `models/agent_rl_vs_ai_final.pkl`
    - Graphe : `graphe/rewards_rl_vs_ai.png`

- **2 – Play RL vs Human**  
  - Vous contrôlez la raquette **droite** (verte) avec les flèches **↑ / ↓**
  - L’agent contrôle la raquette **gauche** (rouge)
  - S’il existe un modèle RL vs AI entraîné, il est chargé automatiquement

- **3 – Train RL vs RL**  
  - Deux agents apprennent l’un contre l’autre
  - Modèles finaux :
    - `models/agent1_rl_vs_rl_final.pkl`
    - `models/agent2_rl_vs_rl_final.pkl`
  - Graphique détaillé : `graphe/rewards_rl_vs_rl.png`

- **4 – Demo trained agent**  
  - Affiche un agent pré‑entraîné en mode RL vs AI
  - Utilise `models/agent_rl_vs_ai_final.pkl` (si présent)

- **5 – Run all modes**  
  - Entraîne successivement :
    - RL vs AI (500 épisodes)
    - RL vs RL (500 épisodes)
  - Produit un graphe comparatif :
    - `graphe/comparison_all_modes.png`

---

## 4. Sauvegarde et chargement des modèles

- Les Q-tables sont sauvegardées automatiquement dans `models/` :
  - `agent_rl_vs_ai_ep100.pkl`, `agent_rl_vs_ai_final.pkl`, etc.
  - `agent1_rl_vs_rl_final.pkl`, `agent2_rl_vs_rl_final.pkl`, etc.
- Pour **réutiliser** un modèle :
  - Option 2 : le mode RL vs Human peut charger un agent RL vs AI déjà entraîné.
  - Option 4 : la démo charge `agent_rl_vs_ai_final.pkl` et affiche ses performances.

Vous pouvez aussi charger un modèle manuellement dans votre propre script :

```python
import agent as ag

agent = ag.Qlearning()
agent.load("models/agent_rl_vs_ai_final.pkl")
print(agent.get_stats())
```

---

## 5. Ce que vous verrez

- **Fenêtre de jeu** (si `render=True`) :
  - Terrain, balle, deux raquettes (vert = RL, rouge = adversaire)
  - Scores affichés en haut
  - Mode de jeu affiché (RL vs AI / Human / RL)
- **Sortie console** :
  - Tous les 100 épisodes : moyenne des récompenses, taux de victoire, taille de Q-table, valeur de ε
- **Graphiques** (dans `graphe/`) :
  - Évolution de la **récompense cumulative** par épisode
  - Moyenne glissante des récompenses
  - Pour RL vs RL : taux de victoire de l’agent 1 dans le temps

---

## 6. Comportement attendu

- **Début d’entraînement** :
  - L’agent explore beaucoup (ε élevé), joue presque au hasard
  - Récompenses très variables, beaucoup de défaites
- **Milieu d’entraînement** :
  - L’agent commence à suivre la balle et à défendre correctement
  - La récompense moyenne augmente progressivement
- **Fin d’entraînement** :
  - L’agent adopte une stratégie plus stable
  - Taux de victoire et récompense moyenne plus élevés

> Pour un apprentissage sérieux, visez **500–1000 épisodes** en mode sans rendu (`render = n`).

---

## 7. Paramètres à ajuster

Dans `main.py` (construction de `GameLearning`) :

- `alpha` : taux d’apprentissage (par défaut 0.5)
- `gamma` : facteur de discount (par défaut 0.95)
- `epsilon` : taux d’exploration initial (par défaut 0.3)
- `eps_decay` : vitesse de décroissance de `epsilon` (par défaut 0.001)

Vous pouvez aussi changer :
- Le **nombre d’épisodes** dans les appels `train_rl_vs_*`.
- Les vitesses de la balle / paddles dans `game.py` pour ajuster la difficulté.

---

## 8. Dépannage

- **`ModuleNotFoundError: No module named 'pygame'`**  
  → `pip install pygame numpy matplotlib`

- **La fenêtre ne se ferme pas**  
  → cliquez sur **X** ou utilisez **Alt+F4** (les événements QUIT sont gérés).

- **Aucun modèle trouvé pour la démo**  
  → lancez d’abord l’option **1** (train RL vs AI) puis l’option **4**.

- **Pas de graphiques**  
  → vérifiez que `matplotlib` est installé et qu’un backend graphique est disponible.

---

Ce README est basé sur les versions actuelles de `agent.py`, `game.py` et `main.py`, ainsi que sur les dossiers `models/` et `graphe/` générés par le code.

# Pong Q-Learning Implementation

Une implémentation complète de Q-learning pour entraîner un agent IA à jouer au Pong avec trois modes de jeu différents.

## Structure du Projet

```
├── agent.py          # Classe Qlearning avec algorithme Q-Learning
├── game.py           # Classe Game - environnement Pong avec pygame
├── main.py           # Script principal d'entraînement et d'évaluation
├── models/           # Modèles Q-tables sauvegardés
├── requirements.txt  # Dépendances Python
├── quick_test.py     # Test rapide de fonctionnalité
└── README.md         # Ce fichier
```

## Installation

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Test rapide (vérifier l'installation)
```bash
python quick_test.py
```

## Utilisation

### Menu Interactif (Recommandé)
```bash
python main.py
```

Le menu suivant s'affichera :

```
======================================================================
PONG Q-LEARNING - MENU PRINCIPAL
======================================================================
1 : Entraîner RL vs AI
2 : Jouer RL vs Human (vous contre l'agent)
3 : Entraîner RL vs RL
4 : Démonstration d'un agent entraîné
5 : Exécuter tous les modes
0 : Quitter
======================================================================
```

### Options du Menu

#### Option 1: Entraîner RL vs AI
- L'agent apprend à jouer contre un adversaire AI simple
- Vous pouvez sauvegarder le modèle entraîné
- Affiche les statistiques et graphiques après l'entraînement

#### Option 2: Jouer RL vs Human
- Vous jouez contre l'agent
- Contrôles: **W/Flèche Haut** = Monter, **S/Flèche Bas** = Descendre
- Vous pouvez charger un agent entraîné
- Fermez la fenêtre pour quitter

#### Option 3: Entraîner RL vs RL
- Deux agents apprennent simultanément en jouant l'un contre l'autre
- Vous pouvez sauvegarder les deux modèles
- Affiche les statistiques et graphiques pour les deux agents

#### Option 4: Démonstration d'un agent entraîné
- Charge un agent entraîné depuis le dossier `models/`
- L'agent joue contre l'AI en mode exploitation pure (pas d'apprentissage)
- Parfait pour voir les performances d'un agent entraîné

#### Option 5: Exécuter tous les modes
- Exécute séquentiellement les modes d'entraînement (1 et 3)
- Utile pour un entraînement complet

### Utilisation en Ligne de Commande

Vous pouvez aussi exécuter directement un mode :

```bash
python main.py 1  # Entraîner RL vs AI
python main.py 2  # Jouer RL vs Human
python main.py 3  # Entraîner RL vs RL
python main.py 4  # Démonstration
python main.py 5  # Exécuter tous les modes
```

## Sauvegarde et Chargement de Modèles

### Sauvegarder un modèle
Lors de l'entraînement (options 1 ou 3), vous pouvez entrer un nom pour sauvegarder le modèle. 
Le modèle sera sauvegardé dans le dossier `models/` avec l'extension `.pkl`.

### Charger un modèle
- **Option 2**: Vous pouvez charger un modèle existant pour jouer contre lui
- **Option 4**: Sélectionnez un modèle pour voir une démonstration

Les modèles sont stockés dans `models/` et peuvent être réutilisés.

## Ce que vous verrez

1. **Fenêtre de jeu**: Gameplay Pong en temps réel
2. **Sortie console**: Mises à jour de progression toutes les 50 épisodes
3. **Statistiques**: Après l'entraînement :
   - Total d'épisodes, étapes, récompenses
   - Taux de victoire, récompense cumulative
   - Taille de la Q-table
4. **Graphiques**: Deux graphiques montrant la progression :
   - Récompense cumulative dans le temps
   - Récompense par épisode

## Résultats Attendus

- **Épisodes précoces**: Taux de victoire faible, exploration aléatoire
- **Épisodes moyens**: Apprentissage de patterns, amélioration des performances
- **Épisodes tardifs**: Meilleure stratégie, taux de victoire plus élevé

*Note: L'apprentissage peut être lent. Pour des résultats plus rapides, augmentez les épisodes (500-1000) ou ajustez les paramètres d'apprentissage.*

## Paramètres

Vous pouvez ajuster les paramètres d'apprentissage dans le code :

- `alpha` (0.0-1.0): Taux d'apprentissage (défaut: 0.5)
- `gamma` (0.0-1.0): Facteur d'actualisation (défaut: 0.9)
- `epsilon` (0.0-1.0): Taux d'exploration (défaut: 0.1)
- `eps_decay`: Taux de décroissance d'epsilon (défaut: 0.0001)
- `episodes`: Nombre d'épisodes d'entraînement (défaut: 200)

## Dépannage

### Problème: "ModuleNotFoundError: No module named 'pygame'"
**Solution**: Installer pygame
```bash
pip install pygame numpy matplotlib
```

### Problème: Fenêtre de jeu ne se ferme pas
**Solution**: Cliquez sur le X ou appuyez sur Alt+F4. La boucle de jeu vérifie les événements QUIT.

### Problème: Aucun modèle trouvé
**Solution**: Entraînez d'abord un agent avec l'option 1 ou 3 et sauvegardez le modèle.

### Problème: Graphiques n'apparaissent pas
**Solution**: 
- Vérifiez que matplotlib fonctionne
- Sur certains systèmes, vous pourriez avoir besoin: `pip install tkinter`

## Fichiers

- `agent.py` - Implémentation de l'agent Q-learning
- `game.py` - Jeu Pong avec trois modes
- `main.py` - Script principal avec menu interactif
- `quick_test.py` - Test rapide de fonctionnalité
- `requirements.txt` - Dépendances Python
- `models/` - Dossier pour les modèles sauvegardés

## Guide de Test

Voir `TESTING_GUIDE.md` pour des instructions de test détaillées.
