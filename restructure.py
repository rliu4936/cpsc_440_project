import os
import shutil

# Define mappings
moves = {
    "DataHandler.py": "src/data_handler.py",
    "IndicatorSignals.py": "src/indicator_signals.py",
    "SignalGenerator.py": "src/signal_generator.py",
    "MetaModel.py": "src/metamodel.py",
    "TripleBarrierLabeler.py": "src/triple_barrier_labeler.py",
    "sma_backtest.py": "experiments/sma_backtest.py",
}

# Files matching pattern to move into cache/
cache_extensions = (".json", ".pkl")

# Files matching pattern to move into plots/
plot_extensions = (".png",)

# Create folders if needed
for folder in ["src", "experiments", "cache", "plots"]:
    os.makedirs(folder, exist_ok=True)

# Move individual files
for old_name, new_name in moves.items():
    if os.path.exists(old_name):
        shutil.move(old_name, new_name)
        print(f"Moved {old_name} → {new_name}")

# Move cache files
for filename in os.listdir("."):
    if filename.endswith(cache_extensions) and filename.startswith("results_cache"):
        shutil.move(filename, os.path.join("cache", filename))
        print(f"Moved {filename} → cache/")

# Move plot files
for filename in os.listdir("."):
    if filename.endswith(plot_extensions):
        shutil.move(filename, os.path.join("plots", filename))
        print(f"Moved {filename} → plots/")

print("\n✅ Done reorganizing project structure!")