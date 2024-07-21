import unittest
from tests.interface import Test
from scripts.pathways import get_kegg_organism, retrieve_all_kegg_pathways, retrieve_all_go_pathways, retrieve_all_msigdb_pathways


class PathwayTest(Test):
    pass


class KeggTest(Test):
    
    def test_organism_name(self):
        assert get_kegg_organism('human') == 'hsa'
        assert get_kegg_organism('homo sapiens') == 'hsa'
        assert get_kegg_organism('zebrafish') == 'dre'
        assert get_kegg_organism('fish') == 'dre'
        assert not get_kegg_organism('hippogriff')

    def test_pathway_retrieval(self):
        pathways = retrieve_all_kegg_pathways('empedobacter brevis', subset=15)
        assert len(pathways) > 3


class GoTest(Test):
    
    def test_pathway_retrieval(self):
        pathways = retrieve_all_go_pathways('human')
        assert len(pathways) > 5000


class MsigdbTest(Test):
    
    def test_pathway_retrieval(self):
        pass
        # pathways = retrieve_all_msigdb_pathways('human')
        # assert len(pathways) > 5000


if __name__ == '__main__':
    unittest.main()
