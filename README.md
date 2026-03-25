# LMStudio LLM Stress Tester

A Python script to stress test local LLM endpoints served via LMStudio or any OpenAI-compatible API.

## Features

- **Context Window Test**: Finds maximum effective context length (up to 500k tokens)
- **Rate Limiting Tests**: Burst and sustained rate testing
- **Parallelism Test**: Concurrent request testing with optional shared memory mode
- **Streaming Test**: Token-by-token streaming performance
- **Error Handling**: Tests malformed requests, edge cases, and prompt injection
- **Memory Stability**: Long-running tests (up to 30 minutes)
- **Deliberation Test**: Multi-agent adversarial deliberation coherence and robustness
- **GPU Monitoring**: Collects GPU metrics if available
- **Progress Indicators**: Visual feedback for long-running tests

## Requirements

- Python 3.8+
- LMStudio server running (or any OpenAI-compatible endpoint)
- Optional: `psutil` for GPU monitoring
- **Virtual environment**: The script will automatically create and run in a virtual environment for isolation. You can also manually activate a venv before running.

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

Note: The script will automatically create a virtual environment in `.venv/` if not already running in one. This ensures isolation from your system Python packages.

## Configuration

Edit `config.json` to customize test parameters:

```json
{
  "endpoint": "http://localhost:1234/v1",
  "max_test_duration_seconds": 1800,
  "tests": {
    "context_window": {
      "min_tokens": 100,
      "max_tokens": 500000,
      "step_tokens": 2000
    },
    "parallelism": {
      "enabled": true,
      "max_workers": 32,
      "shared_memory": true,
      "shared_memory_turns": 5
    },
    "deliberation": {
      "enabled": true,
      "num_turns": 20,
      "topic": "Should our company invest in AI research?",
      "adversarial_turns": 5,
      "context_switch_turns": 3
    },
    "memory_stability": {
      "enabled": true,
      "duration_seconds": 1800
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
Tests multiple concurrent workers to find optimal concurrency level. Optional shared memory mode simulates agents contributing to the same conversation history.

### Streaming Test
Compares streaming vs non-streaming responses, measures first token latency.

### Error Handling Test
Tests server response to malformed requests, edge cases, and prompt injection attacks.

### Deliberation Test
Simulates multi-agent adversarial debates (CEO/Board member harness) to evaluate model's suitability for agentic frameworks. Measures coherence, adversarial robustness, and context retention.

### Memory Stability Test
Runs for up to 30 minutes to detect memory leaks or performance degradation. Configurable duration.

### GPU Monitoring
Collects GPU utilization, memory usage, and temperature if available.

## License

MIT