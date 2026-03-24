# LMStudio LLM Stress Tester

A Python script to stress test local LLM endpoints served via LMStudio or any OpenAI-compatible API.

## Features

- **Context Window Test**: Finds maximum effective context length
- **Rate Limiting Tests**: Burst and sustained rate testing
- **Parallelism Test**: Concurrent request testing
- **Streaming Test**: Token-by-token streaming performance
- **Error Handling**: Tests malformed requests and edge cases
- **Memory Stability**: Long-running tests (up to 5 minutes)
- **GPU Monitoring**: Collects GPU metrics if available

## Requirements

- Python 3.8+
- LMStudio server running (or any OpenAI-compatible endpoint)
- Optional: `psutil` for GPU monitoring

## Usage

```bash
# Basic test with default settings
python stress_tester.py

# Custom configuration
python stress_tester.py --config my_config.json

# Specific tests only
python stress_tester.py --tests context_window,parallelism

# Different endpoint
python stress_tester.py --endpoint http://192.168.1.100:1234/v1

# Disable specific tests
python stress_tester.py --disable memory_stability

# Set max duration
python stress_tester.py --max-duration 180
```

## Configuration

Edit `config.json` to customize test parameters:

```json
{
  "endpoint": "http://localhost:1234/v1",
  "tests": {
    "context_window": {
      "min_tokens": 100,
      "max_tokens": 200000,
      "step_tokens": 2000
    }
  }
}
```

## Output

The script generates:
1. **Terminal Report**: Visual ANSI-formatted output
2. **JSON File**: Detailed results with timestamps
3. **CSV File**: Summary statistics

Results are saved in the current directory with timestamps.

## Test Descriptions

### Context Window Test
Gradually increases prompt length to find maximum context the model can handle.

### Rate Limiting Tests
- **Burst**: Rapid concurrent requests to find burst capacity
- **Sustained**: Constant request rate over time

### Parallelism Test
Tests multiple concurrent workers to find optimal concurrency level.

### Streaming Test
Compares streaming vs non-streaming responses, measures first token latency.

### Error Handling Test
Tests server response to malformed requests and edge cases.

### Memory Stability Test
Runs for up to 5 minutes to detect memory leaks or performance degradation.

### GPU Monitoring
Collects GPU utilization, memory usage, and temperature if available.

## License

MIT