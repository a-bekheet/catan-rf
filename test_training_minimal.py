#!/usr/bin/env python3
"""
Minimal test of the training system structure without heavy dependencies.
Tests that the code structure is correct.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("🧪 Testing Multi-Agent RL Training System")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print()

# Test 1: Check file structure
print("1. Checking file structure...")
required_files = [
    "train_agents.py",
    "catan_rl/agents/base_rl_agent.py",
    "catan_rl/agents/rllib_ppo_agent.py",
    "catan_rl/agents/torchrl_sac_agent.py",
    "catan_rl/agents/langgraph_llm_agent.py",
]

for file in required_files:
    filepath = Path(file)
    if filepath.exists():
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} - MISSING!")
        sys.exit(1)

print()

# Test 2: Check imports (without heavy dependencies)
print("2. Testing base agent import...")
try:
    from catan_rl.agents.base_rl_agent import BaseRLAgent, RandomAgent, AgentMetrics
    print("  ✓ Base agent imports successful")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

print()

# Test 3: Check RandomAgent instantiation
print("3. Testing RandomAgent instantiation...")
try:
    agent = RandomAgent(agent_id=0, config={'seed': 42})
    print(f"  ✓ Created {agent.agent_name}")
    print(f"  ✓ Agent ID: {agent.agent_id}")
    print(f"  ✓ Framework: {agent.framework_name}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print()

# Test 4: Check if game engine exists
print("4. Checking game engine...")
try:
    from catan_rl.core.game.engine import types
    print("  ✓ Game engine types module found")
except ImportError as e:
    print(f"  ⚠️  Game engine import issue (may need dependencies): {e}")

print()

# Test 5: Check documentation
print("5. Checking documentation...")
docs = [
    "MULTI_AGENT_TRAINING.md",
    "QUICK_START_GUIDE.md",
    "UV_SETUP.md",
]
for doc in docs:
    if Path(doc).exists():
        print(f"  ✓ {doc}")
    else:
        print(f"  ✗ {doc} - MISSING!")

print()
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("✅ Structure Test Complete!")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print()
print("Next steps to fully test:")
print("  1. Install dependencies: uv sync")
print("  2. Run full training test")
print()
