import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from agent.graph_builder import build_graph


def test_ping():
    graph = build_graph()
    out = graph.invoke({"user_input": "мир"})
    assert "Привет" in out["response"]
