# 🌐 Symbiotic Twin – IoT Environmental Telemetry Dataset

---

## 📌 Overview

This dataset contains environmental telemetry data collected from three IoT devices. It is designed for building a **Digital Twin (Symbiotic Twin)** system capable of real-time monitoring, anomaly detection, predictive analytics, and intelligent alert generation.

The dataset includes sensor readings such as gas concentration, humidity, temperature, light detection, and motion status. It contains **405,184 records** with **no missing values**.

---

## 📂 Dataset Structure

Each row represents a sensor reading captured at a specific timestamp from a specific IoT device.

| Column Name | Description |
|-------------|-------------|
| `ts` | Unix timestamp of the reading |
| `device` | Unique device identifier (MAC address format) |
| `co` | Carbon Monoxide concentration |
| `humidity` | Humidity percentage (%) |
| `light` | Boolean indicating light detected |
| `lpg` | Liquefied Petroleum Gas concentration |
| `motion` | Boolean indicating motion detected |
| `smoke` | Smoke level |
| `temp` | Temperature in Fahrenheit |

---

## 📊 Dataset Characteristics

| Property | Value |
|----------|-------|
| Total Rows | 405,184 |
| Total Columns | 9 |
| Missing Values | None |
| Device Count | 3 unique IoT devices |
| Data Type | Time-series environmental telemetry |

---

## 🔍 Data Types

| Column | Type | Notes |
|--------|------|-------|
| `ts` | Numeric | Unix time — should be converted to datetime |
| `device` | Categorical | MAC address format |
| `co`, `humidity`, `lpg`, `smoke`, `temp` | Float | Continuous sensor readings |
| `light`, `motion` | Boolean | Binary state indicators |

---

## 🛠 Preprocessing Steps

Before using the dataset, the following preprocessing is recommended:

### 1️⃣ Convert Timestamp

```python
df["ts"] = pd.to_datetime(df["ts"], unit="s")
```

### 2️⃣ Sort by Time

```python
df = df.sort_values("ts")
```

### 3️⃣ Perform Time-Based Train/Test Split

Since this is time-series data, **avoid random splits**. Use a sequential split instead:

```python
train = df.iloc[:int(0.8 * len(df))]
test  = df.iloc[int(0.8 * len(df)):]
```

---

## 🎯 Use Cases in Symbiotic Twin Project

This dataset supports a wide range of applications within the Symbiotic Twin framework:

- 🔥 **Gas Leakage Detection**
- 🚨 **Smoke Alert System**
- 🌡 **Temperature Monitoring**
- 🧠 **Anomaly Detection**
- 📈 **Predictive Modeling**
- 📊 **Real-Time Dashboard Simulation**
- 🤖 **Digital Twin State Simulation**

---

## 🧠 Digital Twin Concept

In the Symbiotic Twin architecture:

- Each `device` acts as a **physical IoT node**.
- The system maintains a **virtual representation** (Digital Twin) of each node.
- Real-time sensor values continuously **update the twin's internal state**.
- AI models **detect abnormal patterns** in the data stream.
- **Alerts are triggered** based on risk scoring logic.

---

## ⚠️ Example Risk Conditions

| Condition | Risk Type |
|-----------|-----------|
| High `smoke` + High `temp` | 🔥 Fire risk |
| High `co` + `motion` detected | 🚨 Safety alert |
| Sudden `temp` spike | 🌡 Environmental anomaly |

---

## 🚀 Future Extensions

- Add real-time streaming pipeline (**Kafka / MQTT**)
- Deploy ML models for **predictive alerts**
- Containerize using **Docker**
- Implement **CI/CD pipeline**
- Deploy on **Kubernetes** cluster

---

## 📌 Project Context

This dataset is part of the **Symbiotic Twin Framework**, a digital twin system designed to:

- Monitor IoT environments
- Learn environmental behavior
- Predict failures
- Provide intelligent alerts
- Simulate environmental conditions

---

## 📄 License

This dataset is intended for **educational and research purposes** only.