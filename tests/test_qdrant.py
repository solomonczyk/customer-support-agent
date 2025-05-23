def test_qdrant_connection():
    from agent.memory import client
    stats = client.get_collections()
    # просто факт, что вернулся словарь/объект, а не исключение
    assert stats is not None
