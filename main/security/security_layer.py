"""
security/security_layer.py
Security layer: HMAC signing, JWT auth, poisoning detection, rate limiting.
"""

import hashlib, hmac, json, time, base64, secrets, numpy as np
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
from config.logging_config import setup_logger

logger = setup_logger("security")

_HMAC_SECRET  = secrets.token_hex(32)
_JWT_SECRET   = secrets.token_hex(32)
_EDGE_SECRETS: Dict[str, str] = {}


def provision_edge(edge_id: str) -> str:
    secret = secrets.token_hex(16)
    _EDGE_SECRETS[edge_id] = secret
    logger.info(f"[Security] Provisioned {edge_id}")
    return secret


def sign_payload(edge_id: str) -> Tuple[str, int]:
    secret = _EDGE_SECRETS.get(edge_id, _HMAC_SECRET)
    ts     = int(time.time())
    body   = json.dumps({"edge_id": edge_id, "ts": ts}, sort_keys=True)
    sig    = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    return sig, ts


def verify_signature(edge_id: str, signature: str, timestamp: int) -> bool:
    if abs(int(time.time()) - timestamp) > 60:
        logger.warning(f"[Security] Replay from {edge_id}")
        return False
    secret   = _EDGE_SECRETS.get(edge_id, _HMAC_SECRET)
    body     = json.dumps({"edge_id": edge_id, "ts": timestamp}, sort_keys=True)
    expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    valid    = hmac.compare_digest(expected, signature)
    if not valid:
        logger.warning(f"[Security] HMAC failed for {edge_id}")
    return valid


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (4 - len(s) % 4))


def issue_token(edge_id: str, ttl: int = 3600) -> str:
    header  = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64(json.dumps({"sub": edge_id, "iat": int(time.time()), "exp": int(time.time()) + ttl}).encode())
    sig_in  = f"{header}.{payload}"
    sig     = hmac.new(_JWT_SECRET.encode(), sig_in.encode(), hashlib.sha256).hexdigest()
    return f"{header}.{payload}.{sig}"


def verify_token(token: str) -> Tuple[bool, str]:
    try:
        h, p, s = token.split(".")
        expected = hmac.new(_JWT_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, s):
            return False, "invalid signature"
        payload = json.loads(_b64d(p))
        if payload["exp"] < int(time.time()):
            return False, "token expired"
        return True, payload["sub"]
    except Exception as e:
        return False, str(e)


class PoisoningDetector:
    def __init__(self, z_threshold: float = 2.5):
        self.z_threshold   = z_threshold
        self.norm_history: Dict[str, List[float]] = defaultdict(list)

    def compute_norms(self, local_weights: Dict) -> Dict[str, float]:
        norms = {}
        for eid, sd in local_weights.items():
            total = sum(float(t.norm(2).item() ** 2) for t in sd.values())
            norms[eid] = float(np.sqrt(total))
        return norms

    def detect(self, local_weights: Dict) -> Tuple[List[str], Dict[str, float]]:
        norms = self.compute_norms(local_weights)
        for eid, n in norms.items():
            self.norm_history[eid].append(n)
        if len(norms) < 2:
            return [], norms
        values = list(norms.values())
        mean, std = np.mean(values), np.std(values)
        suspicious = []
        if std > 0:
            for eid, norm in norms.items():
                z = abs(norm - mean) / std
                if z > self.z_threshold:
                    suspicious.append(eid)
                    logger.warning(f"[PoisoningDetector] {eid} flagged: z={z:.2f}")
        return suspicious, norms


class RateLimiter:
    def __init__(self, max_per_window: int = 5, window_seconds: int = 60):
        self.max_per_window = max_per_window
        self.window_seconds = window_seconds
        self._buckets: Dict[str, List[float]] = defaultdict(list)

    def allow(self, edge_id: str) -> bool:
        now = time.time()
        self._buckets[edge_id] = [t for t in self._buckets[edge_id] if now - t < self.window_seconds]
        if len(self._buckets[edge_id]) >= self.max_per_window:
            return False
        self._buckets[edge_id].append(now)
        return True


_detector:     Optional[PoisoningDetector] = None
_rate_limiter: Optional[RateLimiter]       = None


def get_detector() -> PoisoningDetector:
    global _detector
    if _detector is None:
        _detector = PoisoningDetector()
    return _detector


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def initialize_security(edge_ids: List[str]) -> Dict[str, str]:
    tokens = {}
    for eid in edge_ids:
        provision_edge(eid)
        tokens[eid] = issue_token(eid)
    logger.info(f"[Security] Initialized for {len(edge_ids)} edges")
    return tokens
