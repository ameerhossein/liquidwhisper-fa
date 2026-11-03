from types import SimpleNamespace

from fastapi.testclient import TestClient

from inference_ui.app import app, discover, state


def test_discover_finds_complete_artifacts_and_ignores_training_files():
    files = [
        'runs/run1/checkpoint-100/adapter_config.json',
        'runs/run1/checkpoint-100/adapter_model.safetensors',
        'runs/run1/adapter_snapshots/step-200/adapter_config.json',
        'runs/run1/adapter_snapshots/step-200/adapter_model.bin',
        'runs/run1/merged/config.json',
        'runs/run1/merged/model.safetensors.index.json',
        'runs/run1/checkpoint-300/adapter_config.json',
        'runs/run1/run.json',
    ]
    assert discover(files) == [
        {'path': 'runs/run1/adapter_snapshots/step-200', 'kind': 'adapter', 'label': 'snapshot'},
        {'path': 'runs/run1/checkpoint-100', 'kind': 'adapter', 'label': 'checkpoint'},
        {'path': 'runs/run1/merged', 'kind': 'merged', 'label': 'merged'},
    ]
    assert discover(['config.json', 'model.safetensors']) == [{'path': '', 'kind': 'merged', 'label': 'merged'}]


def test_discovery_api_and_selection_validation(monkeypatch):
    import huggingface_hub
    class FakeApi:
        def __init__(self, **kwargs):
            pass
        def model_info(self, repo, **kwargs):
            assert repo == 'owner/repo'
            return SimpleNamespace(sha='abc', siblings=[SimpleNamespace(rfilename=f) for f in ('config.json', 'model.safetensors')])
    monkeypatch.setattr(huggingface_hub, 'HfApi', FakeApi)
    client = TestClient(app)
    assert client.get('/').status_code == 200
    assert client.post('/api/discover', json={'repo': 'bad'}).status_code == 400
    response = client.post('/api/discover', json={'repo': 'owner/repo'})
    assert response.json()['artifacts'][0]['path'] == ''
    assert state['revision'] == 'abc'
    assert client.post('/api/select', json={'path': '../wrong'}).status_code == 400
