import os
import json
import time
import torch
import csv
import numpy as np
import subprocess
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gpu_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gpu_monitor")

TRIGGER_FLAG_PATH = "/workspace/megatron2/trigger_checkpoint.flag"

class GPUMetricsCollector:
    """Collects and analyzes GPU metrics for failure prediction"""
    
    def __init__(
        self, 
        log_file: str = 'gpu_metrics_final5.csv',
        interval: int = 10,  # seconds
        alert_thresholds: Dict = None
    ):
        self.log_file = os.path.join('gpu_logs', log_file)
        self.interval = interval
        self.running = False
        self.metrics_history = {}
        self.history_window = 60

        self.alert_thresholds = alert_thresholds or {
            'temperature': 85,
            'memory_utilization': 95,
            'power_fluctuation': 15,
            'errors': 0,
            'gpu_utilization_drop': 30,
            'temperature_rise_rate': 5,
        }

        os.makedirs('gpu_logs', exist_ok=True)

        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'timestamp', 'gpu_id', 'gpu_name', 'driver_version', 
                    'temperature', 'power_draw', 'gpu_utilization', 
                    'memory_utilization', 'memory_used', 'memory_total',
                    'errors', 'failure_risk'
                ])

    def start_collection(self):
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("Started GPU metrics collection")

    def stop_collection(self):
        self.running = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=5)
        logger.info("Stopped GPU metrics collection")

    def _collection_loop(self):
        while self.running:
            try:
                metrics_list = self._collect_metrics()
                for gpu_metrics in metrics_list:
                    gpu_id = gpu_metrics['gpu_id']
                    if gpu_id not in self.metrics_history:
                        self.metrics_history[gpu_id] = []
                    self.metrics_history[gpu_id].append(gpu_metrics)
                    if len(self.metrics_history[gpu_id]) > self.history_window:
                        self.metrics_history[gpu_id] = self.metrics_history[gpu_id][-self.history_window:]

                    failure_risk = self._calculate_failure_risk(gpu_id)
                    gpu_metrics['failure_risk'] = failure_risk
                    self._log_metrics(gpu_metrics)

                    if failure_risk >= 0.10:
                        logger.warning(f"HIGH FAILURE RISK ({failure_risk:.2f}) DETECTED FOR GPU {gpu_id}")
                        # === TRIGGER FILE CREATION ===
                        with open(TRIGGER_FLAG_PATH, "w") as f:
                            f.write("trigger_checkpoint")
                        logger.info(f"Created trigger flag: {TRIGGER_FLAG_PATH}")

                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(5)

    def _collect_metrics(self) -> List[Dict]:
        metrics_list = []
        try:
            # Removed ECC error metrics which might not be supported on all GPUs
            output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=index,name,driver_version,temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], universal_newlines=True)

            for line in output.strip().split('\n'):
                values = [val.strip() for val in line.split(',')]
                power_val = values[4]
                power_numeric = float(power_val.replace('W', '')) if 'W' in power_val else float(power_val)

                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'gpu_id': int(values[0]),
                    'gpu_name': values[1],
                    'driver_version': values[2],
                    'temperature': float(values[3]),
                    'power_draw': power_numeric,
                    'gpu_utilization': float(values[5]),
                    'memory_utilization': float(values[6]),
                    'memory_used': float(values[7]),
                    'memory_total': float(values[8]),
                    'errors': 0,  # Default to 0 since we're not collecting error metrics
                }
                metrics_list.append(metrics)
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
        return metrics_list

    def _calculate_failure_risk(self, gpu_id: int) -> float:
        if gpu_id not in self.metrics_history or not self.metrics_history[gpu_id]:
            return 0.0
        history = self.metrics_history[gpu_id]
        current = history[-1]
        risk_factors = []

        # Temperature risk
        temp = current['temperature']
        temp_risk = min(1.0, max(0.0, (temp - 70) / (self.alert_thresholds['temperature'] - 70)))
        risk_factors.append(('temperature', temp_risk))

        # Memory utilization risk
        mem_util = current['memory_utilization']
        mem_risk = min(1.0, max(0.0, (mem_util - 85) / (self.alert_thresholds['memory_utilization'] - 85)))
        risk_factors.append(('memory', mem_risk))

        # Error risk
        error_risk = 1.0 if current['errors'] > 0 else 0.0
        risk_factors.append(('errors', error_risk))

        # Power fluctuation risk
        power_risk = 0.0
        if len(history) >= 2:
            prev_power = history[-2]['power_draw']
            curr_power = current['power_draw']
            if prev_power > 0:
                power_change_pct = abs((curr_power - prev_power) / prev_power * 100)
                power_risk = min(1.0, power_change_pct / self.alert_thresholds['power_fluctuation'])
        risk_factors.append(('power', power_risk))

        # GPU utilization drop risk
        util_risk = 0.0
        if len(history) >= 10:
            baseline_util = np.mean([h['gpu_utilization'] for h in history[:-5]])
            current_util = current['gpu_utilization']
            if baseline_util > 50:
                util_drop_pct = max(0, (baseline_util - current_util))
                util_risk = min(1.0, util_drop_pct / self.alert_thresholds['gpu_utilization_drop'])
        risk_factors.append(('utilization_drop', util_risk))

        # Temperature rise rate risk
        temp_rise_risk = 0.0
        if len(history) >= 6:
            temps = [h['temperature'] for h in history[-6:]]
            time_window_minutes = (self.interval * 6) / 60
            if time_window_minutes > 0:
                temp_rise_rate = (temps[-1] - temps[0]) / time_window_minutes
                if temp_rise_rate > 0:
                    temp_rise_risk = min(1.0, temp_rise_rate / self.alert_thresholds['temperature_rise_rate'])
        risk_factors.append(('temp_rise', temp_rise_risk))

        # Define weights for each risk factor
        weights = {
            'temperature': 0.25,
            'memory': 0.15,
            'errors': 0.25,
            'power': 0.10,
            'utilization_drop': 0.15,
            'temp_rise': 0.10
        }

        # Calculate the weighted total risk
        total_risk = sum(weights[name] * risk for name, risk in risk_factors if name in weights)

        if total_risk > 0.15:
            risk_details = ', '.join([f"{name}: {risk:.2f}" for name, risk in risk_factors])
            logger.warning(f"GPU {gpu_id} failure risk: {total_risk:.2f} ({risk_details})")

        return total_risk

    def _log_metrics(self, metrics: Dict):
        try:
            with open(self.log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    metrics['timestamp'],
                    metrics['gpu_id'],
                    metrics['gpu_name'],
                    metrics['driver_version'],
                    metrics['temperature'],
                    metrics['power_draw'],
                    metrics['gpu_utilization'],
                    metrics['memory_utilization'],
                    metrics['memory_used'],
                    metrics['memory_total'],
                    metrics['errors'],
                    metrics['failure_risk']
                ])
        except Exception as e:
            logger.error(f"Error logging metrics to CSV: {e}")

if __name__ == "__main__":
    metrics_collector = GPUMetricsCollector(interval=5)
    try:
        metrics_collector.start_collection()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        metrics_collector.stop_collection()
        print("Stopped monitoring")