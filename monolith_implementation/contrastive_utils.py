import numpy as np
import tensorflow as tf

def jitter(ts, sigma=0.03):
    """Ajoute du bruit gaussien à une série temporelle."""
    return ts + np.random.normal(loc=0., scale=sigma, size=ts.shape)

def scaling(ts, scaling_factor_range=(0.9, 1.1)):
    """Multiplie une série temporelle par un facteur d'échelle aléatoire."""
    scaling_factor = np.random.uniform(low=scaling_factor_range[0], high=scaling_factor_range[1])
    return ts * scaling_factor

def time_masking(ts, mask_ratio=0.2):
    """Masque une portion aléatoire d'une série temporelle."""
    ts_len = len(ts)
    mask_len = int(ts_len * mask_ratio)
    mask_start = np.random.randint(0, ts_len - mask_len)
    masked_ts = ts.copy()
    masked_ts[mask_start:mask_start + mask_len] = 0
    return masked_ts

def generate_contrastive_pairs_batch(inputs, batch_size=32, augmentation_fn=jitter):
    """
    Génère dynamiquement un batch de paires (ancre, positive, négative) pour l'apprentissage contrastif.
    Args:
        inputs: dict contenant les entrées du modèle (doit inclure 'technical_input', 'embeddings_input', etc.)
        batch_size: taille du batch
        augmentation_fn: fonction d'augmentation à appliquer pour la vue positive
    Returns:
        batch_anchor, batch_pos, batch_neg (dicts pour chaque vue)
    """
    n = inputs["technical_input"].shape[0]
    idx_anchor = np.random.choice(n, batch_size, replace=False)
    idx_neg = np.random.choice(n, batch_size, replace=True)
    # S'assurer que l'ancre et la négative ne sont pas identiques
    idx_neg = np.where(idx_neg == idx_anchor, (idx_neg + 1) % n, idx_neg)

    def make_batch(idx_arr, augmented=False):
        batch = {
            "technical_input": np.stack([
                augmentation_fn(inputs["technical_input"][i]) if augmented else inputs["technical_input"][i]
                for i in idx_arr
            ]),
            "embeddings_input": inputs["embeddings_input"][idx_arr],
            "mcp_input": inputs["mcp_input"][idx_arr],
            "instrument_input": inputs["instrument_input"][idx_arr]
        }
        return batch

    batch_anchor = make_batch(idx_anchor, augmented=False)
    batch_pos = make_batch(idx_anchor, augmented=True)
    batch_neg = make_batch(idx_neg, augmented=False)
    return batch_anchor, batch_pos, batch_neg

def info_nce_loss(anchor, positive, negative, temperature=0.1):
    """
    Calcule la perte InfoNCE pour un batch d'embeddings.
    Args:
        anchor: np.array (batch, dim)
        positive: np.array (batch, dim)
        negative: np.array (batch, dim)
        temperature: float
    Returns:
        loss: float
    """
    # Normalisation L2
    anchor = anchor / np.linalg.norm(anchor, axis=1, keepdims=True)
    positive = positive / np.linalg.norm(positive, axis=1, keepdims=True)
    negative = negative / np.linalg.norm(negative, axis=1, keepdims=True)
    # Similarités cosinus
    pos_sim = np.sum(anchor * positive, axis=1) / temperature
    neg_sim = np.sum(anchor * negative, axis=1) / temperature
    # Logits et labels
    logits = np.stack([pos_sim, neg_sim], axis=1)
    labels = np.zeros(anchor.shape[0], dtype=np.int32)
    # Softmax cross-entropy
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(anchor.shape[0]), labels] + 1e-8))
    return loss

def tf_info_nce_loss(anchor, positive, negative, temperature=0.1):
    """
    Version TensorFlow de la perte InfoNCE pour boucle d'entraînement personnalisée.
    Args:
        anchor: tf.Tensor (batch, dim)
        positive: tf.Tensor (batch, dim)
        negative: tf.Tensor (batch, dim)
        temperature: float
    Returns:
        loss: tf.Tensor (scalaire)
    """
    anchor = tf.math.l2_normalize(anchor, axis=1)
    positive = tf.math.l2_normalize(positive, axis=1)
    negative = tf.math.l2_normalize(negative, axis=1)
    pos_sim = tf.reduce_sum(anchor * positive, axis=1)
    neg_sim = tf.reduce_sum(anchor * negative, axis=1)
    logits = tf.stack([pos_sim, neg_sim], axis=1) / temperature
    labels = tf.zeros(tf.shape(anchor)[0], dtype=tf.int32)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return loss
