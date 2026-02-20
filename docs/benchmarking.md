# Benchmarking Guide

## Running Benchmarks

### Quick Test (1 minute)

```bash
python benchmark.py --duration 60
```

### Full Stress Test (10 minutes)

```bash
python benchmark.py --rpi --duration 600
```

### Custom Videos

```bash
python benchmark.py --videos path/to/video1.mp4 path/to/video2.mp4 --duration 300
```

## What's Measured

| Metric | Source | Interval |
|--------|--------|----------|
| CPU % (per-core) | `psutil.cpu_percent(percpu=True)` | 2s |
| CPU temperature | `/sys/class/thermal/thermal_zone0/temp` | 2s |
| Process RAM (RSS) | `psutil.Process.memory_info()` | 2s |
| System RAM | `psutil.virtual_memory()` | 2s |
| Thread count | `psutil.Process.num_threads()` | 2s |
| Pipeline FPS | EMA-smoothed frame rate | Per frame |
| Frame latency | EMA-smoothed processing time | Per frame |
| YOLO calls | Counter | Cumulative |
| AE calls | Counter | Cumulative |

## Output Files

```
output/
├── benchmark.csv           # Per-sample time series (every 2s)
└── benchmark_summary.txt   # Summary statistics
```

### CSV Columns

```csv
elapsed_s,timestamp,cpu_total_pct,cpu_max_core_pct,ram_used_mb,ram_pct,proc_ram_mb,proc_threads,cpu_temp_c
```

## Real Results (RPi 4B, 10 min)

```
═══════════════════════════════════════════════════════════
  BENCHMARK RESULTS
═══════════════════════════════════════════════════════════
  Duration:        600.7s (10.0 min)  |  264 samples
  
  ── CPU ──
  CPU avg:         60.7%  (±4.4%)
  CPU max (core):  99.0%
  Temp avg:        38.3°C (±0.6°C)
  Temp max:        40.4°C
  
  ── Memory ──
  Process RAM avg: 388.7 MB
  Process RAM max: 391.5 MB
  System RAM:      20.9% used
  
  ── Threads ──
  Threads:         22 (constant)
═══════════════════════════════════════════════════════════
```

## Interpreting Results

### CPU Usage

- **< 60%**: Headroom available — can enable more features
- **60-80%**: Optimal — good utilization without overload
- **> 80%**: Consider enabling backpressure or reducing AE frequency

### Temperature

- **< 50°C**: Excellent cooling
- **50-65°C**: Normal under load
- **65-75°C**: Warm — check fan
- **> 75°C**: Throttling risk — improve cooling

### Memory

- **< 300 MB**: Lightweight — room for growth
- **300-500 MB**: Normal range for this pipeline
- **> 500 MB**: Check for memory leaks (RAM should stabilize after warmup)

### Stability Indicators

✅ **Good**: CPU std dev < 5%, temp std dev < 2°C, RAM stable after 30s  
⚠️ **Warning**: CPU std dev > 10%, temp rising continuously  
❌ **Bad**: RAM growing continuously (memory leak), temp > 75°C
