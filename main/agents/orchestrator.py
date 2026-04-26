"""
agents/orchestrator.py
Multi-Agent Orchestrator for SYMBIOTIC-TWIN.

Four specialized agents run concurrently each federated round:
  - AnalystAgent   : Accuracy trends, convergence, edge divergence
  - AnomalyAgent   : Sensor anomaly detection (fire/gas/temp)
  - PredictorAgent : Next-round accuracy forecast via linear extrapolation
  - SecurityAgent  : Poisoning detection, HMAC failures, trust scoring
"""

import threading
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from config.logging_config import setup_logger

logger = setup_logger("orchestrator")


class AgentStatus(str, Enum):
    IDLE    = "idle"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"


@dataclass
class AgentResult:
    agent_name: str
    round_num:  int
    status:     AgentStatus
    findings:   Dict[str, Any] = field(default_factory=dict)
    alerts:     List[str]      = field(default_factory=list)
    timestamp:  float          = field(default_factory=time.time)


class BaseAgent:
    def __init__(self, name: str):
        self.name   = name
        self.status = AgentStatus.IDLE
        self.logger = setup_logger(f"agent.{name}")

    def run(self, context: Dict[str, Any], round_num: int) -> AgentResult:
        self.status = AgentStatus.RUNNING
        try:
            result = self._execute(context, round_num)
            self.status = AgentStatus.DONE
            return result
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Agent {self.name} error: {e}")
            return AgentResult(agent_name=self.name, round_num=round_num,
                               status=AgentStatus.ERROR, alerts=[str(e)])

    def _execute(self, context: Dict[str, Any], round_num: int) -> AgentResult:
        raise NotImplementedError


# ── Analyst Agent ─────────────────────────────────────────────────────────────

class AnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__("analyst")

    def _execute(self, context, round_num):
        global_acc = context.get("global_acc", [])
        edge_acc   = context.get("edge_acc", {})
        alerts, findings = [], {}

        if len(global_acc) >= 3:
            recent = global_acc[-3:]
            delta  = recent[-1] - recent[0]
            trend  = "improving" if delta > 1.0 else ("regressing" if delta < -1.0 else "plateau")
            findings["trend"]         = trend
            findings["delta_3rounds"] = round(delta, 3)
            if trend == "regressing":
                alerts.append(f"⚠️ Global accuracy declining {delta:.2f}% over last 3 rounds")
            elif trend == "plateau":
                alerts.append("ℹ️ Model has plateaued — cognitive layer should reduce LR")

        if global_acc and edge_acc:
            g = global_acc[-1]
            divs = {}
            for eid, al in edge_acc.items():
                if al:
                    d = abs(al[-1] - g)
                    divs[eid] = round(d, 3)
                    if d > 10.0:
                        alerts.append(f"⚠️ {eid} diverges {d:.1f}% from global — check data distribution")
            findings["edge_divergence"] = divs

        findings["rounds_analysed"] = len(global_acc)
        self.logger.info(f"Analyst round {round_num}: trend={findings.get('trend','N/A')}")
        return AgentResult(agent_name=self.name, round_num=round_num,
                           status=AgentStatus.DONE, findings=findings, alerts=alerts)


# ── Anomaly Agent ─────────────────────────────────────────────────────────────

class AnomalyAgent(BaseAgent):
    THRESHOLDS = {"smoke_critical": 0.10, "co_critical": 0.005,
                  "temp_warning": 90.0,   "lpg_warning": 0.007}

    def __init__(self):
        super().__init__("anomaly")
        self.anomaly_history: List[Dict] = []

    def _execute(self, context, round_num):
        sensor_batch = context.get("sensor_batch", [])
        alerts, findings = [], {"anomalies_detected": 0, "breakdown": {}}
        counts = {"fire_risk": 0, "gas_leak": 0, "temp_spike": 0}

        for r in sensor_batch:
            if r.get("smoke", 0) > self.THRESHOLDS["smoke_critical"] or \
               r.get("co",    0) > self.THRESHOLDS["co_critical"]:
                counts["fire_risk"] += 1
            if r.get("lpg",  0) > self.THRESHOLDS["lpg_warning"]:
                counts["gas_leak"] += 1
            if r.get("temp", 0) > self.THRESHOLDS["temp_warning"]:
                counts["temp_spike"] += 1

        findings["anomalies_detected"] = sum(counts.values())
        findings["breakdown"]          = counts
        findings["batch_size"]         = len(sensor_batch)

        if counts["fire_risk"]:  alerts.append(f"🔥 FIRE RISK: {counts['fire_risk']} readings exceed smoke/CO thresholds")
        if counts["gas_leak"]:   alerts.append(f"⛽ GAS LEAK: {counts['gas_leak']} readings exceed LPG threshold")
        if counts["temp_spike"]: alerts.append(f"🌡️ TEMP SPIKE: {counts['temp_spike']} readings exceed 90°F")

        self.anomaly_history.append({"round": round_num, **counts})
        return AgentResult(agent_name=self.name, round_num=round_num,
                           status=AgentStatus.DONE, findings=findings, alerts=alerts)


# ── Predictor Agent ───────────────────────────────────────────────────────────

class PredictorAgent(BaseAgent):
    def __init__(self):
        super().__init__("predictor")

    def _linear_forecast(self, series, steps=1):
        if len(series) < 2: return None
        n = len(series)
        x_avg = (n - 1) / 2
        y_avg = sum(series) / n
        numer = sum((i - x_avg) * (y - y_avg) for i, y in enumerate(series))
        denom = sum((i - x_avg) ** 2 for i in range(n))
        if denom == 0: return series[-1]
        slope = numer / denom
        return round(series[-1] + slope * steps, 3)

    def _execute(self, context, round_num):
        global_acc = context.get("global_acc", [])
        edge_acc   = context.get("edge_acc", {})
        alerts, findings = [], {}

        pg = self._linear_forecast(global_acc)
        if pg is not None:
            pg = max(0.0, min(100.0, pg))
            findings["predicted_global_acc"] = pg
            if pg < 50.0:
                alerts.append(f"📉 Forecast: global accuracy may drop to {pg:.1f}%")

        ef = {}
        for eid, al in edge_acc.items():
            p = self._linear_forecast(al)
            if p is not None:
                ef[eid] = max(0.0, min(100.0, p))
        findings["edge_forecasts"] = ef

        if ef:
            findings["predicted_best_edge"]  = max(ef, key=ef.get)
            findings["predicted_worst_edge"] = min(ef, key=ef.get)

        return AgentResult(agent_name=self.name, round_num=round_num,
                           status=AgentStatus.DONE, findings=findings, alerts=alerts)


# ── Security Agent ────────────────────────────────────────────────────────────

class SecurityAgent(BaseAgent):
    def __init__(self):
        super().__init__("security")
        self.norm_history: Dict[str, List[float]] = {}

    def _execute(self, context, round_num):
        weight_norms  = context.get("weight_norms", {})
        reputations   = context.get("reputations", {})
        hmac_failures = context.get("hmac_failures", [])
        alerts, findings = [], {"suspicious_edges": [], "norm_deviations": {}}

        import numpy as np
        for eid, norm in weight_norms.items():
            if eid not in self.norm_history:
                self.norm_history[eid] = []
            self.norm_history[eid].append(norm)

        for eid, history in self.norm_history.items():
            if len(history) >= 3:
                avg = sum(history[:-1]) / len(history[:-1])
                if avg > 0:
                    dev = abs(history[-1] - avg) / avg
                    findings["norm_deviations"][eid] = round(dev, 4)
                    if dev > 0.5:
                        findings["suspicious_edges"].append(eid)
                        alerts.append(f"🔐 SECURITY: {eid} norm deviated {dev*100:.1f}% — possible poisoning")

        for eid, score in reputations.items():
            if score < 0.3:
                alerts.append(f"🚨 Low trust: {eid} reputation={score:.3f}")

        for eid in hmac_failures:
            alerts.append(f"🔑 HMAC FAILED: {eid} — payload may be tampered")
            if eid not in findings["suspicious_edges"]:
                findings["suspicious_edges"].append(eid)

        findings["total_alerts"]  = len(alerts)
        findings["hmac_failures"] = hmac_failures
        findings["round_secure"]  = len(alerts) == 0
        return AgentResult(agent_name=self.name, round_num=round_num,
                           status=AgentStatus.DONE, findings=findings, alerts=alerts)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class AgentOrchestrator:
    def __init__(self):
        self.agents = {
            "analyst":   AnalystAgent(),
            "anomaly":   AnomalyAgent(),
            "predictor": PredictorAgent(),
            "security":  SecurityAgent(),
        }
        self.results_log: List[Dict] = []
        self._lock = threading.Lock()

    def run_round(self, context: Dict[str, Any], round_num: int) -> Dict[str, AgentResult]:
        results      = {}
        threads      = []
        results_lock = threading.Lock()

        def _run(name, agent):
            r = agent.run(context, round_num)
            with results_lock:
                results[name] = r

        for name, agent in self.agents.items():
            t = threading.Thread(target=_run, args=(name, agent), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=10)

        with self._lock:
            self.results_log.append({"round": round_num, **results})

        all_alerts = []
        for r in results.values():
            all_alerts.extend(r.alerts)
        if all_alerts:
            logger.warning(f"[Orchestrator] Round {round_num}: {len(all_alerts)} alerts")
            for a in all_alerts:
                logger.warning(f"  {a}")
        return results

    def get_all_alerts(self, round_num: int) -> List[str]:
        for entry in self.results_log:
            if entry.get("round") == round_num:
                alerts = []
                for k, v in entry.items():
                    if isinstance(v, AgentResult):
                        alerts.extend(v.alerts)
                return alerts
        return []

    def get_serializable_results(self, round_num: int) -> Dict:
        for entry in self.results_log:
            if entry.get("round") == round_num:
                out = {}
                for k, v in entry.items():
                    if isinstance(v, AgentResult):
                        out[k] = {"status": v.status, "findings": v.findings, "alerts": v.alerts}
                return out
        return {}


_orchestrator: Optional[AgentOrchestrator] = None

def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
