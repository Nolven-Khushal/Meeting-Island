import sys
sys.path.insert(0, "bench/spk")
sys.path.insert(0, ".")
import diar_test as d
d.run_refined(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
