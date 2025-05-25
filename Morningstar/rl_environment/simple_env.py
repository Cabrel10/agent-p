import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from pathlib import Path
import logging

logger = logging.getLogger("SimpleRLEnv")

class SimpleRLEnv(gym.Env):
    """
    Environnement RL simplifié pour tester l'entraînement d'un agent RL
    sans dépendre de l'encodeur contrastif complexe.

    L'environnement permet à l'agent de choisir parmi 3 actions:
    - Acheter (0)
    - Vendre (1)
    - Ne rien faire (2)
    
    L'agent commence avec un capital initial et peut acheter/vendre un actif unique
    avec des données de prix réelles mais simplifiées.
    """
    
    def __init__(self, data_path, initial_capital=10000.0, transaction_cost_pct=0.001, window_size=10, max_steps=1000, verbose=True):
        super(SimpleRLEnv, self).__init__()
        
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.window_size = window_size
        self.max_steps = max_steps
        self.verbose = verbose
        
        # Chargement des données
        try:
            # Chargement du fichier parquet
            self.df = pd.read_parquet(data_path)
            
            # Sélection d'un seul symbole si plusieurs sont présents
            if 'symbol' in self.df.columns:
                symbols = self.df['symbol'].unique()
                self.df = self.df[self.df['symbol'] == symbols[0]].reset_index(drop=True)
                if self.verbose:
                    logger.info(f"Utilisation d'un seul actif: {symbols[0]}")
                    
            # S'assurer que les colonnes OHLCV sont présentes
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Colonnes manquantes dans les données: {missing_cols}")
                
            # Utiliser uniquement les colonnes nécessaires pour simplifier
            cols_to_use = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            cols_available = [col for col in cols_to_use if col in self.df.columns]
            self.df = self.df[cols_available].reset_index(drop=True)
            
            # Normalisation des features pour l'observation
            self.price_min = self.df['close'].min()
            self.price_max = self.df['close'].max()
            self.volume_min = self.df['volume'].min()
            self.volume_max = self.df['volume'].max()
            
            if self.verbose:
                logger.info(f"Données chargées avec succès: {len(self.df)} entrées")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise
            
        # Définition des espaces d'action et d'observation
        # Actions: 0 (Acheter), 1 (Vendre), 2 (Ne rien faire)
        self.action_space = spaces.Discrete(3)
        
        # Observation: [price_features, position_features]
        # price_features: window_size derniers prix normalisés
        # position_features: [position actuelle, capital disponible normalisé]
        obs_dim = self.window_size + 2
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32)
        
        # Initialisation des variables d'état
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Choix aléatoire d'un point de départ
        start_idx = self.window_size
        end_idx = len(self.df) - self.max_steps
        if end_idx <= start_idx:
            self.current_step = start_idx
        else:
            self.current_step = self.np_random.integers(start_idx, end_idx)
            
        # Initialisation de l'état
        self.capital = self.initial_capital
        self.position = 0.0  # Quantité détenue
        self.steps_done = 0
        self.total_trades = 0
        self.total_profit = 0.0
        
        # Préparer l'observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
        
    def _get_observation(self):
        # Récupérer les window_size derniers prix
        start = self.current_step - self.window_size
        end = self.current_step
        prices = self.df['close'].iloc[start:end].values
        
        # Normalisation des prix
        normalized_prices = (prices - self.price_min) / (self.price_max - self.price_min) * 2 - 1
        
        # Position actuelle et capital normalisés
        current_price = self.df['close'].iloc[self.current_step]
        position_value = self.position * current_price / self.initial_capital * 2 - 1
        capital_normalized = self.capital / self.initial_capital * 2 - 1
        
        # Combiner les features
        observation = np.append(normalized_prices, [position_value, capital_normalized])
        
        return observation.astype(np.float32)
        
    def _calculate_reward(self, old_portfolio_value, new_portfolio_value):
        # Récompense basée sur le changement de valeur du portefeuille
        portfolio_return = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        
        # Pénaliser les transactions excessives
        transaction_penalty = 0.0
        if self.total_trades > 0 and self.steps_done > 0:
            transaction_penalty = 0.0001 * (self.total_trades / self.steps_done)
            
        reward = portfolio_return - transaction_penalty
        
        return reward
        
    def step(self, action):
        # Vérifier si l'épisode est terminé
        if self.steps_done >= self.max_steps:
            truncated = True
            return self._get_observation(), 0.0, False, truncated, {}
            
        # Récupérer le prix actuel
        current_price = self.df['close'].iloc[self.current_step]
        
        # Calculer la valeur du portefeuille avant l'action
        old_portfolio_value = self.capital + self.position * current_price
        
        # Exécuter l'action
        if action == 0:  # Acheter
            if self.capital > 0:
                # Calculer la quantité à acheter (10% du capital disponible)
                amount_to_buy = self.capital * 0.1
                # Calculer les frais
                fee = amount_to_buy * self.transaction_cost_pct
                # Calculer la quantité réelle
                quantity = (amount_to_buy - fee) / current_price
                
                # Mettre à jour l'état
                self.position += quantity
                self.capital -= (amount_to_buy)
                self.total_trades += 1
                
                if self.verbose:
                    logger.debug(f"Achat: {quantity:.6f} @ {current_price:.2f}, Frais: {fee:.2f}")
                    
        elif action == 1:  # Vendre
            if self.position > 0:
                # Calculer la quantité à vendre (50% de la position actuelle)
                quantity = self.position * 0.5
                # Calculer le montant brut
                amount = quantity * current_price
                # Calculer les frais
                fee = amount * self.transaction_cost_pct
                # Calculer le montant net
                net_amount = amount - fee
                
                # Mettre à jour l'état
                self.position -= quantity
                self.capital += net_amount
                self.total_trades += 1
                
                # Calculer le profit
                profit = net_amount - (quantity * current_price)
                self.total_profit += profit
                
                if self.verbose:
                    logger.debug(f"Vente: {quantity:.6f} @ {current_price:.2f}, Profit: {profit:.2f}")
        
        # Passer à l'étape suivante
        self.current_step += 1
        self.steps_done += 1
        
        # Calculer la nouvelle valeur du portefeuille
        new_price = self.df['close'].iloc[self.current_step]
        new_portfolio_value = self.capital + self.position * new_price
        
        # Calculer la récompense
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value)
        
        # Vérifier si l'épisode est terminé
        done = False
        truncated = self.steps_done >= self.max_steps
        
        # Obtenir l'observation mise à jour
        observation = self._get_observation()
        
        # Informations supplémentaires
        info = {
            'portfolio_value': new_portfolio_value,
            'position': self.position,
            'capital': self.capital,
            'price': new_price,
            'total_trades': self.total_trades,
            'total_profit': self.total_profit
        }
        
        return observation, reward, done, truncated, info
        
    def render(self):
        """Simple rendering of the current state"""
        current_price = self.df['close'].iloc[self.current_step]
        portfolio_value = self.capital + self.position * current_price
        
        print(f"Step: {self.steps_done}/{self.max_steps}")
        print(f"Price: {current_price:.2f}")
        print(f"Position: {self.position:.6f}")
        print(f"Capital: {self.capital:.2f}")
        print(f"Portfolio Value: {portfolio_value:.2f}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Total Profit: {self.total_profit:.2f}")
        print("-" * 40)

# Fonction utilitaire pour tester l'environnement
def test_env(env_path, num_episodes=2, render=True):
    """
    Fonction utilitaire pour tester l'environnement avec un agent aléatoire
    
    Args:
        env_path: Chemin vers le fichier de données
        num_episodes: Nombre d'épisodes à exécuter
        render: Afficher l'état à chaque étape
    """
    env = SimpleRLEnv(env_path, verbose=False)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        print(f"Episode {episode+1}/{num_episodes}")
        
        while not (done or truncated):
            action = env.action_space.sample()  # Action aléatoire
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if render and env.steps_done % 100 == 0:
                env.render()
                
        print(f"Episode terminé. Récompense totale: {total_reward:.2f}")
        print(f"Valeur finale du portefeuille: {info['portfolio_value']:.2f}")
        print(f"Profit total: {info['total_profit']:.2f}")
        print(f"Nombre total de transactions: {info['total_trades']}")
        print("=" * 50)
        
    return env

# Fonction pour entraîner un agent avec Stable Baselines 3
def train_agent(env_path, total_timesteps=10000, save_path="models/dqn_simple_agent.zip"):
    """
    Entraîne un agent DQN sur l'environnement SimpleRLEnv
    
    Args:
        env_path: Chemin vers le fichier de données
        total_timesteps: Nombre total d'étapes d'entraînement
        save_path: Chemin où sauvegarder le modèle entraîné
    
    Returns:
        Le modèle DQN entraîné
    """
    # Créer l'environnement
    env = SimpleRLEnv(env_path, verbose=False)
    
    # Créer le modèle DQN
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        verbose=1
    )
    
    # Entraîner le modèle
    model.learn(total_timesteps=total_timesteps)
    
    # Sauvegarder le modèle
    model.save(save_path)
    print(f"Modèle sauvegardé à {save_path}")
    
    return model

# Test du script si exécuté directement
if __name__ == "__main__":
    # Ce code s'exécute uniquement si le script est lancé directement
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Chemin par défaut, à ajuster selon votre structure
        data_path = "ultimate/data/processed/market_features/all_assets_features_merged.parquet"
    
    print(f"Test de l'environnement avec les données: {data_path}")
    
    try:
        # Tester l'environnement avec un agent aléatoire
        env = test_env(data_path, num_episodes=2)
        
        print("Test réussi! L'environnement fonctionne correctement.")
        
        # Entraînement d'un agent (commenter si vous voulez juste tester l'environnement)
        # train_agent(data_path, total_timesteps=5000)
        
    except Exception as e:
        print(f"Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()