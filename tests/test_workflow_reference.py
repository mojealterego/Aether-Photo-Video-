import json
import unittest
from pathlib import Path

PATH=Path(__file__).resolve().parent.parent/"workflows"/"photo400_reference_400mp.gui.json"

class TestReferenceWorkflow(unittest.TestCase):
    def test_static_schema_and_nodes(self):
        graph=json.loads(PATH.read_text(encoding="utf-8"))
        self.assertEqual(graph.get("version"),0.4)
        self.assertEqual(len(graph["nodes"]),51)
        self.assertEqual(len(graph["links"]),73)
        self.assertEqual(len(graph["groups"]),24)
        ids={v["id"] for v in graph["nodes"]}
        for link in graph["links"]:
            self.assertIn(link[1],ids)
            self.assertIn(link[3],ids)

    def test_only_one_active_output(self):
        graph=json.loads(PATH.read_text(encoding="utf-8"))
        writers=[n for n in graph["nodes"] if n["type"]=="Photo400SaveRGB16" and n.get("mode",0)==0]
        self.assertEqual(len(writers),1)

if __name__=="__main__":unittest.main()
