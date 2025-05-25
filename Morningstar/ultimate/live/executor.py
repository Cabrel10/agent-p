# live/executor.py

import os
import asyncio
import pandas as pd
import ccxt  # Ajout pour le client REST
import ccxt.pro as ccxtpro
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
import json  # Ajout pour l'état JSON
from pathlib import Path  # Ajout pour Path

# --- Imports du projet ---
from utils.helpers import load_config  # Supposons qu'une telle fonction existe
from model.architecture.enhanced_hybrid_model import MorningstarModel  # Assurez-vous que le nom est correct

# Renommer ou créer cette fonction/classe pour traiter les données live (OHLCV ici)
from utils.live_preprocessing import LiveDataPreprocessor  # Ou une fonction preprocess_live_data
from live.monitoring import MetricsLogger
from telegram_bot import notify_trade_sync  # Ajout pour les notifications Telegram

# Configuration du Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Constantes ---
RECONNECT_DELAY = 5
ORDER_EXECUTION_DELAY = 0.5


class LiveExecutor:
    """
    Orchestre le trading en direct: connexion exchange, réception données temps réel,
    prédiction modèle, exécution ordres et monitoring.
    """

    def __init__(
        self, config_path: str = "config/config.yaml", exchange_id_override: Optional[str] = None, dry_run: bool = False
    ):
        """
        Initialise l'exécuteur live.

        Args:
            config_path: Chemin vers le fichier de configuration principal.
            exchange_id_override: ID de l'exchange à utiliser, surcharge la config si fourni.
            dry_run: Si True, simule les ordres sans les exécuter réellement.
        """
        logger.info(f"Initialisation de LiveExecutor (Dry Run: {dry_run})...")
        self.dry_run = dry_run
        self.trading_active = True  # Nouvel attribut pour activer/désactiver le trading
        self.max_consecutive_errors = 5  # Nombre maximum d'erreurs consécutives avant pause
        self.consecutive_errors = 0  # Compteur d'erreurs consécutives
        self.error_pause_duration = 300  # Pause de 5 minutes après trop d'erreurs
        self.last_error_time = None  # Horodatage de la dernière erreur
        self.last_ohlcv_received_time = None  # Pour le healthcheck
        self.healthcheck_interval_seconds = 300  # Vérifier toutes les 5 minutes par défaut
        self.max_no_data_interval_seconds = 600  # Alerter si pas de données pendant 10 minutes
        self.last_healthcheck_alert_time = 0  # Pour éviter le spam d'alertes healthcheck
        self.current_pnl = 0.0  # Ajout pour suivre le PnL

        try:
            self.config = load_config(config_path)
            live_config_health = self.config.get("live_trading", {}).get("healthcheck", {})
            self.healthcheck_interval_seconds = live_config_health.get("check_interval_seconds", 300)
            self.max_no_data_interval_seconds = live_config_health.get("no_data_alert_threshold_seconds", 600)
            # Initialiser status_file_path après le chargement de la config
            self.status_file_path = Path(
                self.config.get("live_trading", {}).get("status_file_path", "live_status.json")
            )
            # S'assurer que le répertoire parent existe pour le fichier de statut
            self.status_file_path.parent.mkdir(parents=True, exist_ok=True)
            live_config = self.config.get("live_trading", {})
            exchange_params_config = self.config.get("exchange_params", {})
            model_config = self.config.get("model", {})
            data_pipeline_config = self.config.get("data_pipeline", {})

            # --- Configuration de base ---
            self.exchange_id = (exchange_id_override or live_config.get("default_exchange", "binance")).lower()
            self.symbol = live_config.get("symbol", "BTC/USDT")
            self.timeframe = live_config.get("timeframe", "1m")
            self.use_websocket = live_config.get("websocket", True)

            # --- Paramètres de Trading ---
            self.risk_per_trade_pct: float = live_config.get("risk_per_trade_pct", 0.01)  # Ex: 1%
            self.atr_sl_multiplier: float = live_config.get("atr_sl_multiplier", 1.5)
            self.rr_ratio_tp: float = live_config.get("rr_ratio_tp", 2.0)

            logger.info(
                f"Configuration Live: Exchange={self.exchange_id}, Symbol={self.symbol}, Timeframe={self.timeframe}, WebSocket={self.use_websocket}"
            )
            logger.info(
                f"Paramètres Trading: Risk={self.risk_per_trade_pct*100}%, SL Mult={self.atr_sl_multiplier}, TP Ratio={self.rr_ratio_tp}"
            )

            # --- État de la Position ---
            self.current_position_size: float = 0.0
            self.entry_price: Optional[float] = None
            self.position_side: Optional[str] = None  # 'long' ou None pour l'instant
            self.active_sl_order_id: Optional[str] = None
            self.active_tp_order_id: Optional[str] = None
            self.last_known_balance: Dict[str, Dict[str, float]] = (
                {}
            )  # Ex: {'USDT': {'free': 1000.0, 'used': 0.0, 'total': 1000.0}}

            # --- Composants ---
            # Charger le modèle
            model_path = model_config.get("save_path", "model/saved_model/morningstar_model")
            self.model = MorningstarModel.load_model(model_path)  # Assumer que load_model est une méthode de classe
            logger.info(f"Modèle chargé depuis: {model_path}")

            # Initialiser le preprocessor
            preprocessor_window = data_pipeline_config.get("indicator_window_size", 100)
            self.preprocessor = LiveDataPreprocessor(window_size=preprocessor_window, symbol=self.symbol)

            # Initialiser le logger de métriques (passer la config entière)
            self.metrics = MetricsLogger(config=self.config)  # Assumer que MetricsLogger existe et est importé

            # Initialiser les clients REST et WebSocket
            self.client: Optional[ccxt.Exchange] = None  # Client REST (synchrone ou asynchrone)
            self.ws_client: Optional[ccxtpro.Exchange] = None  # Client WebSocket
            self._init_exchange_clients()  # Peut lever une exception
            logger.info(f"Clients CCXT initialisés pour: {self.exchange_id}")

            if not self.dry_run:
                self._update_balance_sync()

        except FileNotFoundError:
            logger.error(f"Erreur: Fichier de configuration non trouvé à {config_path}")
            raise
        except KeyError as e:
            logger.error(f"Erreur: Clé manquante dans la configuration: {e}")
            raise
        except Exception as e:
            logger.exception(f"Erreur inattendue lors de l'initialisation de LiveExecutor: {e}")
            raise

    def _init_exchange_clients(self):
        logger.info(f"Configuration des clients ccxt/ccxt.pro pour {self.exchange_id}...")
        api_key_env = f"{self.exchange_id.upper()}_API_KEY"
        secret_env = f"{self.exchange_id.upper()}_API_SECRET"
        passphrase_env = f"{self.exchange_id.upper()}_PASSPHRASE"

        api_key = os.getenv(api_key_env)
        secret = os.getenv(secret_env)
        password = os.getenv(passphrase_env)

        if not api_key or not secret:
            logger.error(f"Clés API ({api_key_env}, {secret_env}) non trouvées dans les variables d'environnement.")
            raise ValueError("Clés API manquantes.")

        common_params = self.config.get("exchange_params", {}).get(self.exchange_id, {})
        if "enableRateLimit" not in common_params:
            common_params["enableRateLimit"] = True

        client_config = {"apiKey": api_key, "secret": secret, **common_params}
        ws_client_config = {"apiKey": api_key, "secret": secret, **common_params}

        if password and self.exchange_id in ["kucoin", "bitget"]:  # Exemple d'exchanges nécessitant une passphrase
            client_config["password"] = password
            ws_client_config["password"] = password
        elif not password and self.exchange_id in ["kucoin", "bitget"]:
            logger.error(f"Passphrase ({passphrase_env}) requise mais non trouvée pour {self.exchange_id}.")
            raise ValueError("Passphrase manquante.")

        try:
            rest_cls = getattr(ccxt, self.exchange_id)
            self.client = rest_cls(client_config)
            logger.info(f"Client REST ccxt pour {self.exchange_id} créé.")

            if self.use_websocket:
                ws_cls = getattr(ccxtpro, self.exchange_id)
                self.ws_client = ws_cls(ws_client_config)
                logger.info(f"Client WebSocket ccxt.pro pour {self.exchange_id} créé.")
            else:
                logger.warning("WebSocket désactivé dans la configuration.")
        except AttributeError as e:
            logger.error(f"Exchange '{self.exchange_id}' non supporté par ccxt ou ccxt.pro: {e}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des clients ccxt: {e}")
            raise

    def _update_balance_sync(self):
        if self.dry_run:
            logger.info("[DRY RUN] Mise à jour du solde ignorée.")
            quote_currency = self.symbol.split("/")[1]
            if not self.last_known_balance:
                self.last_known_balance = {quote_currency: {"free": 1000.0, "used": 0.0, "total": 1000.0}}
            return

        if not self.client:
            logger.error("Client CCXT non initialisé, impossible de récupérer le solde.")
            return
        try:
            logger.info("Récupération du solde du compte...")
            balance = self.client.fetch_balance()
            self.last_known_balance = {
                currency: {"free": data.get("free"), "used": data.get("used"), "total": data.get("total")}
                for currency, data in balance.items()
                if isinstance(data, dict)
            }
            quote_currency = self.symbol.split("/")[1]
            logger.info(
                f"Solde mis à jour. Disponible {quote_currency}: {self.last_known_balance.get(quote_currency, {}).get('free', 'N/A')}"
            )
        except Exception as e:
            logger.exception(f"Erreur lors de la récupération du solde: {e}")

    def _calculate_trade_details(self, current_price: float, atr: float) -> Optional[Dict[str, float]]:
        if atr <= 1e-8:
            logger.warning(f"ATR invalide ou nul ({atr}), impossible de calculer les détails du trade.")
            return None
        quote_currency = self.symbol.split("/")[1]
        available_balance = self.last_known_balance.get(quote_currency, {}).get("free")
        if available_balance is None or available_balance <= 0:
            logger.warning(f"Solde disponible insuffisant en {quote_currency} ({available_balance}).")
            return None
        sl_distance_points = atr * self.atr_sl_multiplier
        sl_price = current_price - sl_distance_points
        tp_distance_points = sl_distance_points * self.rr_ratio_tp
        tp_price = current_price + tp_distance_points
        risk_amount_quote = available_balance * self.risk_per_trade_pct
        sl_distance_for_size = current_price - sl_price
        if sl_distance_for_size < 1e-8:
            logger.warning(f"Distance SL trop petite ({sl_distance_for_size}).")
            return None
        order_size_base = risk_amount_quote / sl_distance_for_size
        logger.debug(f"Détails Trade Calculés: Taille={order_size_base:.6f}, SL={sl_price:.4f}, TP={tp_price:.4f}")
        return {"sl_price": sl_price, "tp_price": tp_price, "order_size": order_size_base}

    async def _handle_signal(self, prediction: Dict, current_price: float, atr: float):
        if not self.client:
            logger.error("Client REST non disponible pour gérer le signal.")
            return
        signal = prediction.get("signal")
        if signal is None:
            logger.warning("Signal non trouvé dans la prédiction.")
            return
        signal_label = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(signal, f"UNKNOWN({signal})")
        logger.debug(f"Signal: {signal_label}, État: {self.position_side}, Prix: {current_price:.4f}, ATR: {atr:.4f}")

        if signal == 1 and self.position_side is None:  # BUY signal and FLAT
            trade_details = self._calculate_trade_details(current_price, atr)
            if not trade_details:
                return
            order_size, sl_price, tp_price = (
                trade_details["order_size"],
                trade_details["sl_price"],
                trade_details["tp_price"],
            )
            if order_size <= 0:
                return
            logger.info(
                f"[{'DRY RUN' if self.dry_run else 'LIVE'}] BUY: Taille={order_size:.6f}, SL={sl_price:.4f}, TP={tp_price:.4f}"
            )
            self.metrics.log_trade_attempt(side="buy", symbol=self.symbol, amount=order_size)
            entry_order_info, sl_order_id, tp_order_id, entry_success = None, None, None, False
            try:
                if not self.dry_run:
                    entry_order = await self.client.create_market_buy_order(self.symbol, order_size)
                    entry_order_info = entry_order
                    filled_price = entry_order.get("average", current_price)
                    filled_amount = entry_order.get("filled", order_size)
                    self.entry_price, self.current_position_size = float(filled_price), float(filled_amount)
                else:
                    self.entry_price, self.current_position_size = current_price, order_size
                entry_success = True
                if not self.dry_run:
                    sl_order = await self.client.create_order(
                        self.symbol,
                        "stop_market",
                        "sell",
                        self.current_position_size,
                        params={"stopPrice": sl_price, "reduceOnly": True},
                    )
                    sl_order_id = sl_order["id"]
                    tp_order = await self.client.create_order(
                        self.symbol,
                        "take_profit_market",
                        "sell",
                        self.current_position_size,
                        params={"stopPrice": tp_price, "reduceOnly": True},
                    )
                    tp_order_id = tp_order["id"]
                else:  # Dry run SL/TP IDs
                    sl_order_id, tp_order_id = f"dry_sl_{int(time.time())}", f"dry_tp_{int(time.time())}"
                self.active_sl_order_id, self.active_tp_order_id = sl_order_id, tp_order_id
                self.position_side = "long"
                self.metrics.log_trade_result(success=True, side="buy", symbol=self.symbol, order_info=entry_order_info)
                self.metrics.update_position_size(self.current_position_size, symbol=self.symbol)
                notify_trade_sync(
                    "BUY",
                    self.entry_price,
                    f"Taille: {self.current_position_size:.6f}, SL: {sl_price:.4f}, TP: {tp_price:.4f}.",
                )
                self._update_live_status_file()
            except Exception as e:
                logger.exception(f"Erreur entrée LONG: {e}")
                self.metrics.log_trade_result(
                    success=False, side="buy", symbol=self.symbol, error_type=type(e).__name__
                )
                await self._compensate_failed_entry(entry_success, sl_order_id, tp_order_id)

        elif signal == 2 and self.position_side == "long":  # SELL signal and LONG
            logger.info(f"[{'DRY RUN' if self.dry_run else 'LIVE'}] SELL (Clôture LONG)")
            self.metrics.log_trade_attempt(side="sell", symbol=self.symbol, amount=self.current_position_size)
            original_sl_id, original_tp_id = self.active_sl_order_id, self.active_tp_order_id
            try:
                if original_sl_id and not self.dry_run:
                    await self.client.cancel_order(original_sl_id, self.symbol)
                if original_tp_id and not self.dry_run:
                    await self.client.cancel_order(original_tp_id, self.symbol)
                self.active_sl_order_id, self.active_tp_order_id = None, None
                close_order_info = None
                if not self.dry_run:
                    close_order = await self.client.create_market_sell_order(self.symbol, self.current_position_size)
                    close_order_info = close_order
                    close_price = close_order.get("average", current_price)
                else:
                    close_price = current_price  # Simuler prix de clôture
                pnl = (close_price - self.entry_price) * self.current_position_size if self.entry_price else 0
                self.current_pnl += pnl
                self.metrics.update_pnl(pnl, symbol=self.symbol)
                notify_trade_sync("SELL (CLOSE)", close_price, f"P&L: {pnl:.4f}")
                self._reset_position_state()
                self.metrics.log_trade_result(
                    success=True, side="sell", symbol=self.symbol, order_info=close_order_info
                )
            except Exception as e:
                logger.exception(f"Erreur clôture LONG: {e}")
                self.metrics.log_trade_result(
                    success=False, side="sell", symbol=self.symbol, error_type=type(e).__name__
                )
                # État incertain, pourrait nécessiter une réinitialisation manuelle ou une vérification

    async def _compensate_failed_entry(
        self, entry_success: bool, sl_order_id: Optional[str], tp_order_id: Optional[str]
    ):
        if entry_success and (sl_order_id or tp_order_id):
            logger.warning("Erreur après entrée. Annulation SL/TP.")
            if sl_order_id and not self.dry_run:
                try:
                    await self.client.cancel_order(sl_order_id, self.symbol)
                except Exception as e:
                    logger.error(f"Échec annulation SL ({sl_order_id}): {e}")
            if tp_order_id and not self.dry_run:
                try:
                    await self.client.cancel_order(tp_order_id, self.symbol)
                except Exception as e:
                    logger.error(f"Échec annulation TP ({tp_order_id}): {e}")
            self._reset_position_state()

    def _reset_position_state(self):
        logger.warning("Réinitialisation de l'état de la position à FLAT.")
        self.position_side = None
        self.entry_price = None
        self.current_position_size = 0.0
        self.active_sl_order_id = None
        self.active_tp_order_id = None
        self.metrics.update_position_size(0.0, symbol=self.symbol)
        self._update_live_status_file()

    def _update_live_status_file(self):
        status_data = {
            "timestamp": time.time(),
            "symbol": self.symbol,
            "position_side": self.position_side,
            "entry_price": self.entry_price,
            "current_position_size": self.current_position_size,
            "last_known_balance": self.last_known_balance,
            "active_sl_order_id": self.active_sl_order_id,
            "active_tp_order_id": self.active_tp_order_id,
            "trading_active": self.trading_active,
            "consecutive_errors": self.consecutive_errors,
            "current_pnl": self.current_pnl,  # ← Ajout du PnL actuel
        }
        try:
            with open(self.status_file_path, "w") as f:
                json.dump(status_data, f, indent=4)
            logger.debug(f"Fichier de statut mis à jour : {self.status_file_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du fichier de statut: {e}")

    async def _perform_healthcheck(self):
        current_time = time.time()
        if self.last_ohlcv_received_time:
            time_since_last_data = current_time - self.last_ohlcv_received_time
            # Vérifier si une alerte a déjà été envoyée récemment pour éviter le spam
            if (
                time_since_last_data > self.max_no_data_interval_seconds
                and (current_time - self.last_healthcheck_alert_time) > self.max_no_data_interval_seconds
            ):  # Cooldown simple
                alert_message = (
                    f"⚠️ ALERTE Healthcheck: Aucune nouvelle donnée OHLCV reçue pour {self.symbol} "
                    f"depuis {time_since_last_data:.0f} secondes (seuil: {self.max_no_data_interval_seconds}s)."
                )
                logger.warning(alert_message)
                notify_trade_sync(signal="HEALTH_ALERT", price=0, reasoning=alert_message)
                self.last_healthcheck_alert_time = current_time  # Mettre à jour le temps de la dernière alerte
        else:
            # Si aucune donnée n'a jamais été reçue, initialiser le temps pour la première vérification
            self.last_ohlcv_received_time = current_time
            logger.debug("Healthcheck: Initialisation du temps de réception des données OHLCV.")

    async def healthcheck_loop(self):
        while True:
            await asyncio.sleep(self.healthcheck_interval_seconds)
            await self._perform_healthcheck()

    async def run(self):
        logger.info("Démarrage de la boucle principale de trading...")
        self.metrics.websocket_connection_status.labels(symbol=self.symbol).set(0)
        asyncio.create_task(self.healthcheck_loop())
        self.last_ohlcv_received_time = time.time()  # Initialiser pour le healthcheck

        while True:
            try:
                if self.consecutive_errors >= self.max_consecutive_errors:
                    if self.last_error_time is None or (time.time() - self.last_error_time) < self.error_pause_duration:
                        logger.warning(
                            f"Pause erreurs ({self.consecutive_errors}) pour {self.error_pause_duration/60} min"
                        )
                        await asyncio.sleep(30)
                        continue
                    else:
                        logger.info("Reprise après pause d'erreur")
                        self.consecutive_errors = 0

                if not self.ws_client:  # Vérifier si ws_client est None ou fermé
                    logger.info(f"Initialisation/Réinitialisation du client WebSocket pour {self.exchange_id}...")
                    self._init_exchange_clients()  # S'assure que les clients sont (ré)initialisés
                    if (
                        not self.ws_client and self.use_websocket
                    ):  # Si toujours pas de client WS et qu'on doit l'utiliser
                        logger.error("Échec de l'initialisation du client WebSocket. Nouvel essai...")
                        await asyncio.sleep(RECONNECT_DELAY)
                        continue

                self._update_balance_sync()
                self._update_live_status_file()
                logger.info(f"Écoute des données OHLCV pour {self.symbol}...")

                while True:
                    ohlcv = await self.ws_client.watch_ohlcv(self.symbol, self.timeframe)
                    self.last_ohlcv_received_time = time.time()
                    self.metrics.websocket_connection_status.labels(symbol=self.symbol).set(1)
                    if "timestamp" in ohlcv[-1]:
                        self.metrics.log_websocket_latency(ohlcv[-1]["timestamp"], symbol=self.symbol)

                    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    atr_period = 14  # Devrait être configurable
                    if len(df) >= atr_period:
                        tr1 = df["high"] - df["low"]
                        tr2 = abs(df["high"] - df["close"].shift(1))
                        tr3 = abs(df["low"] - df["close"].shift(1))
                        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                        df["atr"] = tr.rolling(window=atr_period).mean()
                        last_atr = df["atr"].iloc[-1]
                        current_price = df["close"].iloc[-1]
                        try:
                            model_input = self.preprocessor.preprocess_live_data(df)
                            if model_input is not None:
                                raw_predictions = self.model.predict(model_input)
                                final_signal = (
                                    np.argmax(raw_predictions["signal"][0]) if "signal" in raw_predictions else 0
                                )
                                formatted_predictions = {"signal": final_signal}
                                self.metrics.log_prediction(formatted_predictions, symbol=self.symbol)
                                if self.trading_active:
                                    await self._handle_signal(formatted_predictions, current_price, last_atr)
                                else:
                                    logger.info(f"Trading désactivé, signal ignoré: {formatted_predictions}")
                                self.consecutive_errors = 0
                                self._update_live_status_file()
                        except Exception as e:
                            logger.exception(f"Erreur prédiction/gestion signal: {e}")
                            self.metrics.log_error(component="prediction_execution", error_type=type(e).__name__)
                            self.consecutive_errors += 1
                            self.last_error_time = time.time()
                            self._update_live_status_file()
            except (ccxtpro.NetworkError, ccxtpro.ExchangeError, asyncio.CancelledError) as e:
                error_type = (
                    "network_error"
                    if isinstance(e, ccxtpro.NetworkError)
                    else "exchange_error" if isinstance(e, ccxtpro.ExchangeError) else "cancelled_error"
                )
                logger.warning(f"Erreur WebSocket ({error_type}): {e}. Reconnexion...")
                self.metrics.log_error(component="websocket", error_type=error_type)
                if not isinstance(e, asyncio.CancelledError):
                    self.consecutive_errors += 1
                    self.last_error_time = time.time()
                self._update_live_status_file()
                if isinstance(e, asyncio.CancelledError):
                    break
            except Exception as e:
                logger.exception(f"Erreur inattendue boucle principale: {e}. Reconnexion...")
                self.metrics.log_error(component="main_loop", error_type="unexpected")
                self.consecutive_errors += 1
                self.last_error_time = time.time()
                self._update_live_status_file()

            await self.close_ws_client()  # Fermer avant de retenter
            logger.info(f"Attente de {RECONNECT_DELAY}s avant reconnexion...")
            await asyncio.sleep(RECONNECT_DELAY)
            # Pas besoin de réinitialiser les clients ici, la boucle le fera si ws_client est None

    async def close_ws_client(self):
        if self.ws_client:
            logger.info("Fermeture connexion WebSocket...")
            try:
                await self.ws_client.close()
                self.metrics.websocket_connection_status.labels(symbol=self.symbol).set(0)
            except Exception as e:
                logger.error(f"Erreur fermeture WS: {e}")
            finally:
                self.ws_client = None

    async def close(self):
        await self.close_ws_client()
        logger.info("Client REST ccxt ne nécessite pas de fermeture explicite.")
        self.client = None  # Peut aider au garbage collection

    def activate_trading(self):
        if not self.trading_active:
            self.trading_active = True
            logger.info("Trading activé.")
            return True
        return False

    def deactivate_trading(self):
        if self.trading_active:
            self.trading_active = False
            logger.info("Trading désactivé.")
            return True
        return False

    def get_trading_status(self):
        return {
            "active": self.trading_active,
            "consecutive_errors": self.consecutive_errors,
            "max_consecutive_errors": self.max_consecutive_errors,
            "error_pause_active": self.consecutive_errors >= self.max_consecutive_errors,
            "last_error_time": self.last_error_time,
            "current_pnl": self.current_pnl,
        }

    async def process_override_command(
        self, action: str, symbol_override: str, price: Optional[float] = None
    ) -> Tuple[bool, str]:
        logger.info(f"Override: {action} {symbol_override} @ {price if price else 'MARKET'}")
        if symbol_override.upper() != self.symbol.upper():
            return False, f"Override ignoré: Symbole {symbol_override} != {self.symbol}."
        if not self.client:
            return False, "Client CCXT non initialisé."

        action = action.upper()
        market_price = None
        try:
            ticker = await self.client.fetch_ticker(self.symbol)
            market_price = ticker.get("last") or (ticker.get("bid", 0) + ticker.get("ask", 0)) / 2
            if not market_price:
                raise ValueError("Prix du marché non récupérable")
        except Exception as e:
            logger.error(f"Erreur fetch_ticker pour override: {e}")
            return False, f"Erreur fetch_ticker: {str(e)}"

        execution_price: float = price if price is not None else market_price

        current_side = self.position_side
        current_size = self.current_position_size
        current_entry_price = self.entry_price

        try:
            if self.active_sl_order_id and not self.dry_run:
                await self.client.cancel_order(self.active_sl_order_id, self.symbol)
            self.active_sl_order_id = None
            if self.active_tp_order_id and not self.dry_run:
                await self.client.cancel_order(self.active_tp_order_id, self.symbol)
            self.active_tp_order_id = None

            if current_side == "long" and action != "BUY":
                logger.info(f"Override: Clôture LONG {current_size} {self.symbol.split('/')[0]}.")
                if not self.dry_run:
                    close_order = await self.client.create_market_sell_order(self.symbol, current_size)
                    closed_price = close_order.get("average", execution_price)
                    pnl = (closed_price - current_entry_price) * current_size if current_entry_price else 0
                    self.current_pnl += pnl
                    notify_trade_sync("OVERRIDE CLOSE LONG", closed_price, f"P&L: {pnl:.2f}")
                else:
                    closed_price_sim = execution_price
                    pnl = (closed_price_sim - current_entry_price) * current_size if current_entry_price else 0
                    self.current_pnl += pnl
                    logger.info(f"[DRY RUN] Override: Clôture LONG @ {closed_price_sim:.4f}. P&L: {pnl:.2f}")
                self._reset_position_state()

            if action == "BUY":
                quote_currency = self.symbol.split("/")[1]
                available_balance = self.last_known_balance.get(quote_currency, {}).get("free", 0)
                if available_balance <= 0 and not self.dry_run:
                    return False, "Solde insuffisant pour override BUY."

                order_size_override = (available_balance * 0.05) / execution_price if execution_price > 0 else 0.001
                order_size_override = max(order_size_override, 0.0001)  # TODO: Utiliser les limites de l'exchange

                if not self.dry_run:
                    buy_order = await self.client.create_market_buy_order(self.symbol, order_size_override)
                    self.entry_price = buy_order.get("average", execution_price)
                    self.current_position_size = buy_order.get("filled", order_size_override)
                else:
                    self.entry_price = execution_price
                    self.current_position_size = order_size_override
                self.position_side = "long"
                logger.info(f"Override BUY: Taille={self.current_position_size}, Prix={self.entry_price}")
                notify_trade_sync("OVERRIDE BUY", self.entry_price, f"Taille: {self.current_position_size}")
                self._update_live_status_file()
                return True, f"Override BUY exécuté pour {self.symbol}."

            elif action == "SELL":
                if self.position_side is None:
                    return False, "Override SELL ignoré: Pas de position LONG à clôturer."
                # La clôture est déjà gérée si current_side == "long" et action != "BUY" (ce qui inclut SELL)
                return True, "Position LONG clôturée par override SELL."

            elif action == "HOLD":
                msg = (
                    f"Override HOLD: Ordres SL/TP annulés. Position maintenue: {self.position_side}"
                    if self.position_side
                    else "Override HOLD: Aucune position active."
                )
                logger.info(msg)
                self._update_live_status_file()
                return True, msg

            return True, "Commande override traitée."
        except Exception as e:
            logger.error(f"Erreur traitement override: {e}")
            return False, f"Erreur override: {str(e)}"
