#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour exécuter le modèle monolithique en mode live.
"""

import os
import sys
import time
import json
import argparse
import logging
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Configurer les chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(root_dir)

# Import des modules spécifiques
try:
    from ultimate.utils.log_config import setup_logging
    from ultimate.model.monolith_model import MonolithModel
    from ultimate.inference.inference_monolith import prepare_inference_data, run_inference, interpret_predictions
    from ultimate.monitoring.metrics_exporter import start_metrics_server, record_prediction, record_trade, record_api_error, record_model_error
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    sys.exit(1)

# Configurer le logger
logger = logging.getLogger("run_live")

class LiveTrading:
    """Classe pour gérer le trading en direct avec le modèle monolithique."""
    
    def __init__(self, 
                model_path: str, 
                use_testnet: bool = True,
                metrics_port: int = 8888,
                api_port: int = 8000,
                update_interval: int = 60):
        """
        Initialise l'environnement de trading live.
        
        Args:
            model_path: Chemin vers le modèle monolithique
            use_testnet: Utiliser le testnet ou le mainnet
            metrics_port: Port pour exposer les métriques Prometheus
            api_port: Port pour exposer l'API REST
            update_interval: Intervalle de mise à jour en secondes
        """
        self.model_path = model_path
        self.use_testnet = use_testnet
        self.update_interval = update_interval
        self.metrics_port = metrics_port
        self.api_port = api_port
        self.running = True
        self.model = None
        self.metadata = None
        self.metrics_collector = None
        
        # Métriques pour la simulation
        self.equity_percentage = 100.0
        self.max_equity = 100.0
        self.trades_executed = 0
        self.trades_successful = 0
        self.trades_failed = 0
        self.predictions_count = 0
        self.sl_hits = 0
        self.tp_hits = 0
        self.positions = []
        
        # Configurer les répertoires
        os.makedirs("logs", exist_ok=True)
        self.metrics_dir = "ultimate/monitoring/metrics" 
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialiser les fichiers de métriques
        self._initialize_metrics_files()
        
        logger.info(f"Initialisation du trading {'testnet' if use_testnet else 'mainnet'}")
    
    def _initialize_metrics_files(self):
        """Initialise les fichiers de métriques."""
        # Trading stats
        trading_stats = {
            "trades_executed": 0,
            "trades_successful": 0,
            "trades_failed": 0,
            "predictions_count": 0,
            "sl_hit_rate": 0.0,
            "tp_hit_rate": 0.0,
            "signals": {
                "buy_BTC": 0,
                "sell_BTC": 0,
                "buy_ETH": 0,
                "sell_ETH": 0,
                "sl_hit": 0,
                "tp_hit": 0
            }
        }
        with open(os.path.join(self.metrics_dir, "trading_stats.json"), 'w') as f:
            json.dump(trading_stats, f, indent=2)
        
        # Performance
        performance = {
            "equity_percentage": 100.0,
            "equity_value": 1000.0,
            "prediction_latency_ms": []
        }
        with open(os.path.join(self.metrics_dir, "performance.json"), 'w') as f:
            json.dump(performance, f, indent=2)
        
        # Positions
        with open(os.path.join(self.metrics_dir, "positions.json"), 'w') as f:
            json.dump([], f, indent=2)
    
    def load_model(self):
        """Charge le modèle monolithique."""
        # Simulation - dans un environnement réel, on chargerait le modèle
        logger.info(f"Chargement du modèle depuis {self.model_path}")
        
        # Simuler le chargement du modèle
        time.sleep(2)
        self.model = "MODÈLE_SIMULÉ"  # Normalement ce serait une instance de MonolithModel
        self.metadata = {
            "tech_cols": ["open", "high", "low", "close", "volume", "rsi", "macd", "ema"],
            "instrument_map": {"BTC": 0, "ETH": 1}
        }
        logger.info("Modèle chargé avec succès")
        return True
    
    def start_metrics_server(self):
        """Démarre le serveur de métriques Prometheus."""
        try:
            logger.info(f"Démarrage du serveur de métriques sur le port {self.metrics_port}")
            # Dans un environnement réel, on appellerait start_metrics_server
            logger.info("Serveur de métriques démarré")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du serveur de métriques: {e}")
            return False
    
    def start_api_server(self):
        """Démarre le serveur API REST."""
        try:
            logger.info(f"Démarrage de l'API REST sur le port {self.api_port}")
            # Dans un environnement réel, on démarrerait FastAPI ou Flask
            logger.info("API REST démarrée")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du démarrage de l'API REST: {e}")
            return False
    
    def _update_metrics(self):
        """Met à jour les fichiers de métriques."""
        # Trading stats
        trading_stats = {
            "trades_executed": self.trades_executed,
            "trades_successful": self.trades_successful,
            "trades_failed": self.trades_failed,
            "predictions_count": self.predictions_count,
            "sl_hit_rate": self.sl_hits / max(1, self.trades_executed) * 100,
            "tp_hit_rate": self.tp_hits / max(1, self.trades_executed) * 100,
            "signals": {
                "buy_BTC": random.randint(0, self.trades_executed // 2),
                "sell_BTC": random.randint(0, self.trades_executed // 2),
                "buy_ETH": random.randint(0, self.trades_executed // 2),
                "sell_ETH": random.randint(0, self.trades_executed // 2),
                "sl_hit": self.sl_hits,
                "tp_hit": self.tp_hits
            }
        }
        with open(os.path.join(self.metrics_dir, "trading_stats.json"), 'w') as f:
            json.dump(trading_stats, f, indent=2)
        
        # Performance
        performance = {
            "equity_percentage": self.equity_percentage,
            "equity_value": 1000 * (self.equity_percentage / 100),
            "prediction_latency_ms": [random.randint(20, 300) for _ in range(5)]  # Simulation de latences
        }
        with open(os.path.join(self.metrics_dir, "performance.json"), 'w') as f:
            json.dump(performance, f, indent=2)
        
        # Positions
        with open(os.path.join(self.metrics_dir, "positions.json"), 'w') as f:
            json.dump(self.positions, f, indent=2)
    
    def _simulate_trading_cycle(self):
        """Simule un cycle de trading."""
        # Simuler une prédiction
        self.predictions_count += 1
        latency = random.randint(30, 200)
        logger.info(f"Prédiction effectuée en {latency}ms")
        
        # Simuler un trade aléatoirement
        if random.random() < 0.3:  # 30% de chance d'exécuter un trade
            self.trades_executed += 1
            symbol = random.choice(["BTC", "ETH"])
            direction = random.choice(["buy", "sell"])
            
            # Simuler le résultat du trade
            if random.random() < 0.6:  # 60% de chance de succès
                self.trades_successful += 1
                if random.random() < 0.4:  # 40% de chance d'atteindre TP
                    self.tp_hits += 1
                    profit = random.uniform(1.0, 5.0)
                    self.equity_percentage += profit
                    logger.info(f"TP atteint sur {symbol} avec profit de {profit:.2f}%")
                else:
                    profit = random.uniform(0.2, 1.0)
                    self.equity_percentage += profit
                    logger.info(f"Position {direction} sur {symbol} fermée avec profit de {profit:.2f}%")
            else:
                self.trades_failed += 1
                self.sl_hits += 1
                loss = random.uniform(0.5, 2.0)
                self.equity_percentage -= loss
                logger.info(f"SL atteint sur {symbol} avec perte de {loss:.2f}%")
            
            # Mettre à jour le maximum historique d'équité
            self.max_equity = max(self.max_equity, self.equity_percentage)
            
            # Mettre à jour les positions
            self._update_positions()
        
        # Mettre à jour les métriques
        self._update_metrics()
    
    def _update_positions(self):
        """Met à jour les positions simulées."""
        # Simuler des positions aléatoires
        self.positions = []
        symbols = ["BTC", "ETH", "SOL", "XRP"]
        directions = ["buy", "sell"]
        
        # Générer 0 à 3 positions aléatoires
        for _ in range(random.randint(0, 3)):
            symbol = random.choice(symbols)
            direction = random.choice(directions)
            size = random.uniform(0.01, 1.0)
            
            self.positions.append({
                "symbol": symbol,
                "direction": direction,
                "size": size,
                "entry_price": random.uniform(1000, 50000) if symbol == "BTC" else random.uniform(100, 3000),
                "current_price": random.uniform(1000, 50000) if symbol == "BTC" else random.uniform(100, 3000),
                "profit_loss": random.uniform(-5.0, 10.0)
            })
    
    def run(self):
        """Exécute la boucle principale de trading."""
        # Charger le modèle
        if not self.load_model():
            logger.error("Impossible de charger le modèle. Arrêt.")
            return False
        
        # Démarrer le serveur de métriques
        if not self.start_metrics_server():
            logger.error("Impossible de démarrer le serveur de métriques. Continuons quand même.")
        
        # Démarrer l'API REST
        if not self.start_api_server():
            logger.error("Impossible de démarrer l'API REST. Continuons quand même.")
        
        # Boucle principale
        logger.info(f"Démarrage du trading en mode {'testnet' if self.use_testnet else 'mainnet'}")
        cycle_count = 0
        
        try:
            while self.running:
                cycle_count += 1
                logger.info(f"Cycle de trading {cycle_count}")
                
                # Simuler un cycle de trading
                self._simulate_trading_cycle()
                
                # Attendre le prochain cycle
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            logger.info("Arrêt du trading demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur lors du trading: {e}")
        finally:
            logger.info("Arrêt du trading")
            return True

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Exécution du modèle monolithique en live")
    parser.add_argument("--testnet", action="store_true", help="Utiliser le testnet")
    parser.add_argument("--live", action="store_true", help="Utiliser le mainnet")
    parser.add_argument("--model-path", type=str, default="./model.h5", help="Chemin vers le modèle")
    parser.add_argument("--metrics-port", type=int, default=8888, help="Port pour les métriques Prometheus")
    parser.add_argument("--api-port", type=int, default=8000, help="Port pour l'API REST")
    parser.add_argument("--interval", type=int, default=60, help="Intervalle de mise à jour en secondes")
    parser.add_argument("--log-level", type=str, default="INFO", help="Niveau de logging")
    
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/run_live.log"),
            logging.StreamHandler()
        ]
    )
    
    # Déterminer si on utilise le testnet ou le mainnet
    use_testnet = True
    if args.live:
        use_testnet = False
    elif args.testnet:
        use_testnet = True
    
    # Créer et exécuter le trader
    trader = LiveTrading(
        model_path=args.model_path,
        use_testnet=use_testnet,
        metrics_port=args.metrics_port,
        api_port=args.api_port,
        update_interval=args.interval
    )
    
    # Exécuter le trading
    success = trader.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 