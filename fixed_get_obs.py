def _get_obs(self):
    try:
        current_data_idx = min(self.current_step, len(self.data) - 1)
        market_row = self.data.iloc[current_data_idx]
        
        # Vérification critique: s'assurer que technical_cols_for_market_features est défini
        if not self.technical_cols_for_market_features:
            logger.error("CRITICAL _get_obs: technical_cols_for_market_features est vide!")
            expected_market_features_len = self.observation_space.shape[0] - (1 + len(self.assets))
            market_features_raw = np.zeros(expected_market_features_len, dtype=np.float32)
        else:
            # Vérifier que toutes les colonnes requises sont présentes dans le DataFrame
            missing_cols = [col for col in self.technical_cols_for_market_features if col not in market_row.index]
            if missing_cols:
                logger.error(f"Colonnes manquantes dans le DataFrame: {missing_cols}")
                logger.info(f"Colonnes disponibles: {market_row.index.tolist()[:10]}...")
                # Créer un array de zéros pour les colonnes manquantes
                expected_market_features_len = len(self.technical_cols_for_market_features)
                market_features_raw = np.zeros(expected_market_features_len, dtype=np.float32)
            else:
                try:
                    # S'assurer qu'on utilise EXACTEMENT les colonnes définies dans technical_cols_for_market_features
                    market_features_raw = market_row[self.technical_cols_for_market_features].values.astype(np.float32)
                    
                    # Vérification de compatibilité avec le scaler
                    if self.use_encoder_for_obs and self.scaler is not None:
                        if hasattr(self.scaler, 'n_features_in_'):
                            if self.scaler.n_features_in_ != len(self.technical_cols_for_market_features):
                                logger.error(
                                    f"INCOHÉRENCE DANS GET_OBS: Nombre de features marché ({len(self.technical_cols_for_market_features)}) "
                                    f"différent de ce que le scaler attend ({self.scaler.n_features_in_})."
                                )
                except KeyError as e:
                    logger.error(f"KeyError in _get_obs selecting technical_cols: {e}. Columns in market_row: {market_row.index.tolist()[:10]}")
                    expected_market_features_len = len(self.technical_cols_for_market_features)
                    market_features_raw = np.zeros(expected_market_features_len, dtype=np.float32)

        # Remplacer les NaN/Inf par des zéros
        if np.isnan(market_features_raw).any() or np.isinf(market_features_raw).any():
            market_features_raw = np.nan_to_num(market_features_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Préparer les features du portefeuille
        # Normaliser le capital entre 0 et 2 par rapport au capital initial (permet de dépasser 1 si profit)
        normalized_capital = np.clip(self.capital / (self.initial_capital if self.initial_capital > 1e-9 else 1.0), 0, 2)
        
        # Positions pour chaque actif (normalisées par rapport au capital initial)
        asset_positions_normalized = np.zeros(len(self.assets), dtype=np.float32)
        for i, asset_id in enumerate(self.assets):
            if asset_id in self.positions:
                asset_price = self._get_asset_price(asset_id)
                if asset_price > 1e-9:
                    position_value = self.positions[asset_id]["qty"] * asset_price
                    asset_positions_normalized[i] = np.clip(position_value / (self.initial_capital if self.initial_capital > 1e-9 else 1.0), -2, 2)
        
        portfolio_state_features = np.concatenate([[normalized_capital], asset_positions_normalized]).astype(np.float32)
        
        # 3. Appliquer l'encodeur si nécessaire
        final_obs_market_features = market_features_raw  # Valeur par défaut si on n'utilise pas l'encodeur

        if self.use_encoder_for_obs:
            try:
                # Vérification supplémentaire de compatibilité
                if self.scaler is not None and hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != market_features_raw.shape[0]:
                    logger.critical(f"Incompatibilité critique: Le scaler attend {self.scaler.n_features_in_} features, mais {market_features_raw.shape[0]} sont fournies.")
                    # Fallback vers les features brutes
                    self.use_encoder_for_obs = False
                else:
                    # Normaliser avec le scaler
                    market_features_scaled = self.scaler.transform(market_features_raw.reshape(1, -1))
                    market_features_scaled = np.nan_to_num(market_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Encoder avec le modèle Keras
                    encoded_output = self.encoder(market_features_scaled, training=False)
                    if hasattr(encoded_output, 'numpy'):
                        final_obs_market_features = encoded_output[0].numpy()
                    else:
                        final_obs_market_features = np.array(encoded_output[0], dtype=np.float32)
                    
                    final_obs_market_features = np.nan_to_num(final_obs_market_features, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception as e_enc_runtime:
                logger.critical(f"ERREUR CRITIQUE _get_obs: Échec de l'encodage: {e_enc_runtime}.", exc_info=self.verbose_env)
                expected_encoded_len = self.observation_space.shape[0] - len(portfolio_state_features)
                final_obs_market_features = np.zeros(expected_encoded_len, dtype=np.float32)
        
        # 4. Construire l'observation finale
        final_obs = np.concatenate([final_obs_market_features, portfolio_state_features]).astype(np.float32)

        # 5. Vérifications finales
        if final_obs.shape[0] != self.observation_space.shape[0]:
            logger.critical(
                f"BUG CRITIQUE _get_obs: Shape de l'observation final ({final_obs.shape}) "
                f"!= espace attendu ({self.observation_space.shape}). "
                f"MarketFeat: {final_obs_market_features.shape}, PortfolioFeat: {portfolio_state_features.shape}"
            )
            raise ValueError(f"Bug critique: Incohérence de la forme de l'observation. Attendu {self.observation_space.shape}, obtenu {final_obs.shape}")
        
        if np.isnan(final_obs).any() or np.isinf(final_obs).any():
            logger.error(f"CRITICAL _get_obs: NaN/Inf DANS L'OBSERVATION FINALE step {self.current_step}! Remplacement.")
            final_obs = np.nan_to_num(final_obs, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return final_obs
        
    except Exception as e_obs:
        logger.critical(f"Erreur générale fatale dans _get_obs step {self.current_step}: {e_obs}", exc_info=True)
        return np.zeros(self.observation_space.shape, dtype=np.float32)
