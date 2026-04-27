import unittest
import os
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from knowledge_graph import KnowledgeArchitect
from swarm_io import SwarmIO

class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        SwarmIO.ensure_directories()
        self.architect = KnowledgeArchitect()
        
    def test_clean_text(self):
        raw = "Hello   World\n\nThis is a    test."
        clean = self.architect._clean_text(raw)
        self.assertEqual(clean, "Hello World This is a test.")

    def test_ingest_markdown(self):
        # Create dummy md
        md_content = "# Header\n\nSome **bold** text."
        md_path = "test_doc.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
            
        try:
            text = self.architect.ingest_file(md_path)
            self.assertIn("Header", text)
            self.assertIn("bold", text)
        finally:
            if os.path.exists(md_path):
                os.remove(md_path)

    @patch('knowledge_graph.GeminiCLIWrapper')
    def test_build_graph_mock(self, MockWrapper):
        # Mock the Gemini response
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "nodes": {
                "root": {"label": "Test Domain", "summary": "Root", "type": "concept", "children": ["c1"]},
                "c1": {"label": "Sub Concept", "summary": "Sub", "type": "concept", "children": []}
            }
        })
        
        # Setup mock instance
        mock_instance = MockWrapper.return_value
        # Use AsyncMock for async methods
        mock_instance.generate_content_async = AsyncMock(return_value=mock_response)
        
        self.architect.model = mock_instance 
        
        # Run build
        text = "This is some dummy text."
        filename = self.architect.build_graph(text, "TestDomain")
        
        # Verify file created
        filepath = os.path.join(SwarmIO.KG_DIR, filename)
        self.assertTrue(os.path.exists(filepath))
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.assertIn("nodes", data)
            self.assertIn("root", data["nodes"])

    def test_query_graph(self):
        # Create a dummy graph manually
        domain = "QueryTest"
        filename = self.architect._create_dummy_graph(domain)
        
        # Query root
        result = self.architect.query_graph(filename, "root")
        self.assertEqual(result["label"], domain)
        self.assertEqual(len(result["children"]), 2)
        self.assertEqual(result["children"][0]["id"], "c1")

    @patch('knowledge_graph.GeminiCLIWrapper')
    def test_enrich_system_prompt_mock(self, MockWrapper):
        # Setup mock
        mock_instance = MockWrapper.return_value
        mock_response = MagicMock()
        mock_response.text = "You are a Quantum Physics expert."
        # Use AsyncMock
        mock_instance.generate_content_async = AsyncMock(return_value=mock_response)
        
        self.architect.model = mock_instance
        
        # Create dummy graph
        graph_path = self.architect._create_dummy_graph("Physics")
        
        # Test
        enhanced = self.architect.enrich_system_prompt("Helpful assistant", graph_path)
        self.assertEqual(enhanced, "You are a Quantum Physics expert.")
        
        # Cleanup
        os.remove(os.path.join(SwarmIO.KG_DIR, graph_path))

if __name__ == '__main__':
    unittest.main()
