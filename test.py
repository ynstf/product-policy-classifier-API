import unittest
from fastapi.testclient import TestClient
from app import app

class TestServer(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_server_runs(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"welcome", response.content.lower())

if __name__ == '__main__':
    unittest.main()

