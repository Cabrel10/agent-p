def _get_obs(self):
    """Obtenir l'observation actuelle (features du marché + état du portefeuille).

    Si l'encodeur est activé (self.use_encoder_for_obs=True), les features du marché sont encodées 
    avant d'être concaténées avec les features du portefeuille.

    Returns:
        np.ndarray: Vecteur d'observation pour l'agent RL
    """
    try:
        # 1. Récupérer les features du marché (colonnes techniques spécifiées)
        current_data_idx = min(self.current_step, len(self.data) - 1)
        
        # Ceci suppose que self.data.iloc[current_data_idx] donne une ligne de features de marché agrégées
        # ou des features pour un actif principal. Si les données sont longues (par actif par ligne), cela nécessite une refonte
        # pour obtenir des features pour tous les actifs ou une vue d'ensemble du marché.
        market_row = self.data.iloc[current_data_idx]
        
        if not self.technical_cols_for_market_features:  # Ne devrait pas se produire si __init__ est robuste
            logger.error("CRITICAL _get_obs: technical_cols_for_market_features est vide!")
            # Fallback vers des zéros correspondant à la partie des features de marché de l'espace d'observation
            expected_market_features_len = self.observation_space.shape[0] - (1 + len(self.assets))
            market_features_raw = np.zeros(expected_market_features_len, dtype=np.float32)
        else:
            try:
                market_features_raw = market_row[self.technical_cols_for_market_features].values.astype(np.float32)
            except KeyError as e:
                logger.error(f"KeyError dans _get_obs lors de la sélection des colonnes techniques: {e}. Colonnes dans market_row: {market_row.index.tolist()[:10]}")
                expected_market_features_len = self.observation_space.shape[0] - (1 + len(self.assets))
                market_features_raw = np.zeros(expected_market_features_len, dtype=np.float32)

        if np.isnan(market_features_raw).any() or np.isinf(market_features_raw).any():
            # logger.warning(f"NaN/Inf dans market_features_raw step {self.current_step}. Remplacement par 0.")
            market_features_raw = np.nan_to_num(market_features_raw, nan=0.0, posinf=0.0, neginf=0.0)

        normalized_capital = np.clip(self.capital / (self.initial_capital if self.initial_capital > 1e-9 else 1.0), 0, 2)
        asset_positions_normalized = np.zeros(len(self.assets), dtype=np.float32)
        for i, asset_id in enumerate(self.assets):
            if asset_id in self.positions:
                asset_price = self._get_asset_price(asset_id)
                if asset_price > 1e-9:
                    position_value = self.positions[asset_id]["qty"] * asset_price
                    asset_positions_normalized[i] = np.clip(position_value / (self.initial_capital if self.initial_capital > 1e-9 else 1.0), -2, 2)
        
        portfolio_state_features = np.concatenate([[normalized_capital], asset_positions_normalized]).astype(np.float32)
        
        final_obs_market_features = market_features_raw  # Par défaut si l'encodeur n'est pas utilisé

        if self.use_encoder_for_obs:  # Drapeau défini dans __init__
            try:
                # Ceci suppose que self.scaler et self.encoder sont valides et que les dimensions correspondent en raison des vérifications __init__
                market_features_scaled = self.scaler.transform(market_features_raw.reshape(1, -1))
                market_features_scaled = np.nan_to_num(market_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Logique de prédiction/appel Keras (simplifiée par rapport à l'original)
                encoded_output = self.encoder(market_features_scaled, training=False)  # TF EagerTensor
                if hasattr(encoded_output, 'numpy'):
                    final_obs_market_features = encoded_output[0].numpy()
                else:  # Fallback si ce n'est pas un tenseur avec .numpy() par ex. TF plus ancien ou tableau numpy direct
                    final_obs_market_features = np.array(encoded_output[0], dtype=np.float32)
                
                final_obs_market_features = np.nan_to_num(final_obs_market_features, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception as e_enc_runtime:
                logger.critical(f"ERREUR CRITIQUE _get_obs: Échec de l'encodage (use_encoder_for_obs=True): {e_enc_runtime}. "
                               "Retour de zéros pour les features marché encodées.", exc_info=self.verbose_env)
                expected_encoded_len = self.observation_space.shape[0] - len(portfolio_state_features)
                final_obs_market_features = np.zeros(expected_encoded_len, dtype=np.float32)

        final_obs = np.concatenate([final_obs_market_features, portfolio_state_features]).astype(np.float32)

        if final_obs.shape[0] != self.observation_space.shape[0]:
            logger.critical(
                f"BUG CRITIQUE _get_obs: Shape de l'observation final ({final_obs.shape}) "
                f"!= espace attendu ({self.observation_space.shape}). "
                f"MarketFeat: {final_obs_market_features.shape}, PortfolioFeat: {portfolio_state_features.shape}, use_encoder: {self.use_encoder_for_obs}"
            )
            # Cela ne devrait pas se produire avec la nouvelle logique __init__. Si c'est le cas, c'est un bug profond.
            # Forcer un crash pourrait être préférable à un padding/troncature silencieux.
            raise ValueError(f"Bug critique: Incohérence de la forme de l'observation. Attendu {self.observation_space.shape}, obtenu {final_obs.shape}")
        
        if np.isnan(final_obs).any() or np.isinf(final_obs).any():
            logger.error(f"CRITICAL _get_obs: NaN/Inf DANS L'OBSERVATION FINALE step {self.current_step}! Remplacement.")
            final_obs = np.nan_to_num(final_obs, nan=0.0, posinf=max(np.finfo(np.float32).max, 1e10), neginf=min(np.finfo(np.float32).min, -1e10))  # Remplacer par de grands nombres finis
        
        return final_obs
        
    except Exception as e_obs:
        logger.critical(f"Erreur générale fatale dans _get_obs step {self.current_step}: {e_obs}", exc_info=True)
        # Retourner un vecteur de zéros correspondant à l'espace d'observation pour éviter le crash, mais c'est mauvais.
        return np.zeros(self.observation_space.shape, dtype=np.float32)
